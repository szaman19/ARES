"""End-to-end proof that ARES, Fusion, and DecisionProbe run with ZERO SQL or DuckDB.

Tests:
1. Standalone pure-Python `task_type="verify"` execution in `ares.solve(..., backend=None)`
   enforcing Invariant I4 (context isolation), Invariant I2 (taint guard), and Invariant I6
   (content-hash deduplication across rounds).
2. Complete non-SQL `Domain` (`CodebaseConcurrencyDomain` over JSON/AST call & lock graphs)
   running end-to-end through `run_investigation_loop()` on multi-clue, decoy, and
   under-determined cases without touching DuckDB or Parquet.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ares import NoteStore, ProcessorResult, TaskGraph, TaskSpec, solve
from contracts import (
    CostEstimate,
    DecisionQuestion,
    Domain,
    ExecutionBackend,
    GroundTruth,
    GuardValidationError,
    HypothesisSpec,
    InMemoryVerifierRegistry,
    InvestigationBoard,
    NeedleSpec,
    Note,
    VerifierRegistry,
    VerifierResult,
)
from fusion import LoopConfig, run_investigation_loop
from models import DecisionProbeClient
from models.gemini_provider import ModelTelemetry


# ---------------------------------------------------------------------------
# 1. Pure-Python `ares.solve(backend=None)` Standalone Verifier Tests
# ---------------------------------------------------------------------------


def test_standalone_python_verifiers_run_without_any_backend() -> None:
    """`task_type='verify'` works natively with `backend=None` (zero SQL/DuckDB)."""
    seen_inputs: list[dict[str, Any]] = []

    def _trusted_check(params: dict[str, Any]) -> VerifierResult:
        seen_inputs.append(dict(params))
        ok = params.get("smiles") == "c1ccccc1"
        return VerifierResult(
            verifier_name="check_aromatic_ring",
            passed=ok,
            certification="CERTIFIED" if ok else "UNCERTIFIED",
            evidence={"smiles": params.get("smiles"), "aromatic": ok},
            row_count=1 if ok else 0,
            summary="Aromatic ring verified." if ok else "Not benzene.",
        )

    def _untrusted_heuristic(params: dict[str, Any]) -> VerifierResult:
        return VerifierResult(
            verifier_name="llm_heuristic",
            passed=True,
            certification="CERTIFIED",  # Must be downgraded to UNCERTIFIED by ARES taint guard!
            evidence={"smiles": params.get("smiles")},
            row_count=1,
            summary="Heuristic guess",
        )

    reg = InMemoryVerifierRegistry()
    reg.register("check_aromatic_ring", _trusted_check, trusted=True)
    reg.register("llm_heuristic", _untrusted_heuristic, trusted=False)

    store = NoteStore()
    graph = TaskGraph()

    r1 = solve(
        tasks=[
            TaskSpec(
                task_id="t_trusted",
                task_type="verify",
                verifier_name="check_aromatic_ring",
                verifier_params={"smiles": "c1ccccc1"},
                context={"board_secret": "must_never_leak_into_verifier"},
                requested_under=0,
            ),
            TaskSpec(
                task_id="t_untrusted",
                task_type="verify",
                verifier_name="llm_heuristic",
                verifier_params={"smiles": "c1ccccc1"},
                requested_under=0,
            ),
        ],
        graph=graph,
        backend=None,
        registry=reg,
        note_store=store,
    )

    assert r1.errors == {}
    assert r1.usage.completed_tasks == 2
    assert r1.usage.verifiers_executed == 2
    assert r1.usage.certified_notes == 1
    assert r1.usage.uncertified_notes == 1
    assert seen_inputs == [{"smiles": "c1ccccc1"}]  # Invariant I4 verified!

    # Round 2: Re-submitting identical verifier+params deduplicates automatically
    r2 = solve(
        tasks=[
            TaskSpec(
                task_id="t_trusted_dup",
                task_type="verify",
                verifier_name="check_aromatic_ring",
                verifier_params={"smiles": "c1ccccc1"},
                requested_under=1,
            )
        ],
        graph=graph,
        backend=None,
        registry=reg,
        note_store=store,
    )
    assert r2.graph.nodes["t_trusted_dup"].status == "skipped_duplicate"
    assert r2.usage.skipped_duplicate_tasks == 1
    assert len(store) == 2


# ---------------------------------------------------------------------------
# 2. Complete Non-SQL Domain + Pluggable Graph Backend through 8-Step Loop
# ---------------------------------------------------------------------------

MODULES = ("SessionPool", "TokenCache", "OrderQueue", "AuditWriter")
CODE_PIVOTAL_FIELDS = [
    "public_entry_reachable",
    "lock_order_inversion",
    "deterministic_race_repro",
]


class _CodeGraphBackend:
    """Tiny non-SQL `ExecutionBackend` querying a JSON codebase analysis graph."""

    def __init__(self, workspace_dir: str) -> None:
        self.workspace_dir = Path(workspace_dir)

    def _load(self) -> dict[str, dict[str, bool]]:
        return json.loads((self.workspace_dir / "code_graph.json").read_text(encoding="utf-8"))

    def validate_query(self, query: str) -> str:
        cleaned = query.strip()
        if not cleaned.startswith("inspect:"):
            raise GuardValidationError("CodeGraph magnet must start with 'inspect:'.")
        if ".." in cleaned or ";" in cleaned:
            raise GuardValidationError("Forbidden token in CodeGraph magnet.")
        return cleaned

    def compute_content_hash(
        self,
        query: str,
        *,
        hypothesis_family: str = "",
        verifier_name: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        from ares import canonical_content_hash

        return canonical_content_hash(
            query.strip().lower(),
            hypothesis_family=hypothesis_family,
            verifier_name=verifier_name,
            params=params,
        )

    def estimate_cost(self, query: str) -> CostEstimate:
        self.validate_query(query)
        target = query.split(":", 1)[1].strip()
        units = 10 if target != "*" else 100
        return CostEstimate(estimated_units=units, estimated_bytes=units * 64)

    def execute_query(
        self,
        query: str,
        *,
        params: dict[str, Any] | None = None,
        requested_under: int = 0,
        hypothesis_family: str = "",
        content_hash: str = "",
        note_store: NoteStore | None = None,
    ) -> tuple[dict[str, Any], Note]:
        del params
        norm = self.validate_query(query)
        c_hash = content_hash or self.compute_content_hash(
            norm, hypothesis_family=hypothesis_family
        )
        graph_data = self._load()
        prop = norm.split(":", 1)[1].strip()
        matched = [m for m, attrs in graph_data.items() if attrs.get(prop)]
        nid = note_store.next_note_id() if note_store is not None else "n_cg_1"
        note = Note(
            note_id=nid,
            certification="UNCERTIFIED",
            sources=[f"code_graph:{prop}"],
            hypothesis_family=hypothesis_family,
            query=norm,
            content_hash=c_hash,
            requested_under=requested_under,
            summary=f"CodeGraph inspect:{prop} matched {matched}",
            evidence={"matched_modules": matched},
            row_count=len(matched),
        )
        if note_store is not None:
            note_store.append(note)
        return {"matched_modules": matched}, note

    def execute_verifier(
        self,
        registry: VerifierRegistry,
        verifier_name: str,
        *,
        params: dict[str, Any] | None = None,
        requested_under: int = 0,
        hypothesis_family: str = "",
        query: str = "",
        content_hash: str = "",
        note_store: NoteStore | None = None,
    ) -> tuple[VerifierResult, Note]:
        fn = registry.get(verifier_name)
        c_hash = content_hash or self.compute_content_hash(
            query,
            hypothesis_family=hypothesis_family,
            verifier_name=verifier_name,
            params=params,
        )
        res: VerifierResult = fn(self._load(), dict(params or {}))
        cert = res.certification if res.passed else "UNCERTIFIED"
        nid = note_store.next_note_id() if note_store is not None else "n_ver"
        note = Note(
            note_id=nid,
            certification=cert,
            sources=[f"verifier:{verifier_name}"],
            verifier_name=verifier_name,
            hypothesis_family=hypothesis_family,
            query=query,
            content_hash=c_hash,
            requested_under=requested_under,
            summary=res.summary,
            evidence=dict(res.evidence),
            row_count=res.row_count,
        )
        if note_store is not None:
            note_store.append(note)
        return res, note


class CodebaseConcurrencyDomain:
    """Non-SQL `Domain` auditing concurrency bugs across static AST, lock graphs, and repro tests."""

    name = "codebase_concurrency_audit"

    def generate(self, out_dir: str, size_gb: float = 0.001, seed: int = 1) -> None:
        del size_gb, seed
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        base = {
            m: {"reachable": False, "lock_cycle": False, "race_repro": False}
            for m in MODULES
        }
        (root / "code_graph.json").write_text(json.dumps(base), encoding="utf-8")

    def plant(self, out_dir: str, needle: NeedleSpec) -> GroundTruth:
        root = Path(out_dir)
        data = json.loads((root / "code_graph.json").read_text(encoding="utf-8"))
        t = needle.target_entity
        data[t] = {
            "reachable": needle.missing_signal != "reachable",
            "lock_cycle": needle.missing_signal != "lock_cycle",
            "race_repro": needle.missing_signal != "race_repro",
        }
        if needle.decoy_entity:
            # High-salience 2-of-3 decoy: reachable + lock_cycle warning, but race_repro is False!
            data[needle.decoy_entity] = {
                "reachable": True,
                "lock_cycle": True,
                "race_repro": False,
            }
        (root / "code_graph.json").write_text(json.dumps(data), encoding="utf-8")
        verified = needle.category != "under_determined"
        return GroundTruth(
            case_id=needle.case_id,
            category=needle.category,
            primary_target=t if verified else None,
            expected_decisions={
                "buggy_module": t if verified else "none_verified",
                "sufficiently_verified": "yes" if verified else "no",
            },
            is_sufficiently_verified=verified,
            expected_status="CERTIFIED" if verified else "UNCERTIFIED",
            pivotal_fields=list(CODE_PIVOTAL_FIELDS),
            decoy_target=needle.decoy_entity,
        )

    def needle_menu(self) -> list[NeedleSpec]:
        return [
            NeedleSpec(
                case_id="code_decoy_session_vs_cache",
                category="look_alike_decoy",
                scenario_type="deadlock_race",
                target_entity="SessionPool",
                decoy_entity="TokenCache",
            ),
            NeedleSpec(
                case_id="code_ud_missing_repro",
                category="under_determined",
                scenario_type="unreproduced_lock_warning",
                target_entity="OrderQueue",
                missing_signal="race_repro",
            ),
        ]

    def environment_description(self) -> str:
        return "Codebase analysis graph with properties: reachable, lock_cycle, race_repro."

    def magnet_rules(self) -> str:
        return "Emit `inspect:<property>` where property is reachable, lock_cycle, or race_repro."

    def create_backend(self, workspace_dir: str) -> ExecutionBackend:
        return _CodeGraphBackend(workspace_dir)

    def pivotal_fields(self) -> list[str]:
        return list(CODE_PIVOTAL_FIELDS)

    def required_signal_fields(self) -> tuple[str, ...]:
        return tuple(CODE_PIVOTAL_FIELDS)

    def hypothesis_families(self) -> list[str]:
        return ["entry_reachability", "lock_graph_cycles", "stress_repro"]

    def decision_questions(self) -> list[DecisionQuestion]:
        return [
            DecisionQuestion(
                question_id="buggy_module",
                question="Which module has a verified concurrency bug?",
                options=[*MODULES, "none_verified"],
                question_type="multiple_choice",
            ),
            DecisionQuestion(
                question_id="sufficiently_verified",
                question="Are reachability, lock cycle, and deterministic race repro all verified?",
                options=["yes", "no"],
                question_type="yes_no",
            ),
        ]

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        def _make_prop_verifier(vname: str, prop: str):
            def _v(graph_data: dict[str, dict[str, bool]], params: dict[str, Any]) -> VerifierResult:
                del params
                hits = [m for m, attrs in graph_data.items() if attrs.get(prop)]
                ok = len(hits) > 0
                return VerifierResult(
                    verifier_name=vname,
                    passed=ok,
                    certification="CERTIFIED" if ok else "UNCERTIFIED",
                    evidence={"candidates": hits},
                    row_count=len(hits),
                    summary=f"{vname}: {hits}",
                )

            return _v

        registry.register(
            "verify_reachability",
            _make_prop_verifier("verify_reachability", "reachable"),
            trusted=True,
        )
        registry.register(
            "verify_lock_inversion",
            _make_prop_verifier("verify_lock_inversion", "lock_cycle"),
            trusted=True,
        )
        registry.register(
            "verify_race_repro",
            _make_prop_verifier("verify_race_repro", "race_repro"),
            trusted=True,
        )

    def extract_field_patches_from_note(self, note: Note) -> dict[str, Any] | None:
        mapping = {
            "verify_reachability": "public_entry_reachable",
            "verify_lock_inversion": "lock_order_inversion",
            "verify_race_repro": "deterministic_race_repro",
        }
        fname = mapping.get(note.verifier_name or "")
        if not fname:
            return None
        cands = note.evidence.get("candidates", [])
        if not cands:
            return {fname: "none"}
        return {fname: cands[0] if len(cands) == 1 else list(cands)}

    def score(self, board: dict[str, Any], truth: GroundTruth) -> dict[str, Any]:
        ans = board["decision"]["answer"]
        return {
            "correct": (ans == truth.primary_target)
            if truth.is_sufficiently_verified
            else (ans == "none_verified"),
            "decoy_avoided": ans != truth.decoy_target,
        }


class _MockNonSQLProvider:
    def make_propose_processor(self, **kwargs: Any) -> Any:
        del kwargs

        def _proc(spec: TaskSpec, dep_outputs: dict[str, Any]) -> ProcessorResult:
            del dep_outputs
            fam = (spec.context or {}).get("hypothesis_family", "entry_reachability")
            prop_map = {
                "entry_reachability": "inspect:reachable",
                "lock_graph_cycles": "inspect:lock_cycle",
                "stress_repro": "inspect:race_repro",
            }
            return ProcessorResult(
                output=[
                    HypothesisSpec(
                        query=prop_map.get(fam, "inspect:reachable"),
                        hypothesis_family=fam,
                        expected_signal=f"Check {fam}",
                        requested_under=spec.requested_under,
                    )
                ],
                prompt_tokens=50,
                completion_tokens=20,
                model_calls=1,
            )

        return _proc

    def propose_board_patch(
        self, *, new_notes: list[Note], **kwargs: Any
    ) -> tuple[dict[str, Any], ModelTelemetry]:
        del kwargs
        updates: dict[str, Any] = {}
        q_notes = [n for n in new_notes if n.verifier_name is None and n.row_count > 0]
        if q_notes:
            mods = q_notes[0].evidence.get("matched_modules", [])
            if mods:
                updates["exploratory_hint"] = {
                    "value": mods[0],
                    "sources": [q_notes[0].note_id],
                    "artifact_ids": [],
                }
        return {"updates": updates, "rationale": "Non-SQL synthesis"}, ModelTelemetry(model_calls=1)

    def propose_followup_tasks(
        self, *, board: InvestigationBoard, **kwargs: Any
    ) -> tuple[list[HypothesisSpec], ModelTelemetry]:
        del kwargs
        return (
            [
                HypothesisSpec(
                    query="inspect:race_repro",
                    hypothesis_family="followup",
                    expected_signal="Recheck repro",
                    requested_under=board.board_version,
                )
            ],
            ModelTelemetry(model_calls=1),
        )


class _MockGenericDecisionClient:
    def __init__(self, required_fields: Sequence[str]) -> None:
        self._req = tuple(required_fields)
        self.decisions = SimpleNamespace(create=self._create)

    def _create(self, *, model: str, input: Any, questions: dict[str, dict[str, Any]]) -> Any:
        assert model == "gemini-decision-1"
        raw = input[0]["text"] if isinstance(input, list) else input
        fields = json.loads(raw)["fields"]

        primary_qid = next(qid for qid, q in questions.items() if q["type"] == "choice")
        opts = [o for o in questions[primary_qid]["criteria"] if o != "none_verified"]

        winner = None
        if all(
            fields.get(r, {}).get("status") == "certified"
            and fields.get(r, {}).get("value") not in ("unknown", "none")
            for r in self._req
        ):
            for cand in opts:
                if all(
                    fields[r]["value"] == cand
                    or (isinstance(fields[r]["value"], list) and cand in fields[r]["value"])
                    for r in self._req
                ):
                    winner = cand
                    break

        probs = {o: 0.0 for o in questions[primary_qid]["criteria"]}
        if winner:
            probs[winner] = 0.96
            probs["none_verified"] = 0.04
            choice = winner
            noul = 0.97
        else:
            probs["none_verified"] = 0.95
            if opts:
                probs[opts[0]] = 0.05
            choice = "none_verified"
            noul = 0.03

        return SimpleNamespace(
            answers={
                primary_qid: SimpleNamespace(choice=choice, confidence=0.95, probabilities=probs),
                "sufficiently_verified": SimpleNamespace(noul=noul, confidence=0.96),
            },
            usage=SimpleNamespace(total_input_tokens=80, total_tokens=80),
        )


def test_non_sql_domain_runs_full_investigation_loop_end_to_end(tmp_path: Path) -> None:
    domain = CodebaseConcurrencyDomain()
    assert isinstance(domain, Domain)
    assert isinstance(domain.create_backend(str(tmp_path)), ExecutionBackend)

    # Case 1: Decoy disambiguation (SessionPool has all 3 signals; TokenCache decoy has 2/3)
    decoy_dir = tmp_path / "code_decoy"
    domain.generate(str(decoy_dir))
    truth_decoy = domain.plant(str(decoy_dir), domain.needle_menu()[0])

    probe = DecisionProbeClient(client=_MockGenericDecisionClient(CODE_PIVOTAL_FIELDS))
    res_decoy = run_investigation_loop(
        domain,
        str(decoy_dir),
        provider=_MockNonSQLProvider(),  # type: ignore[arg-type]
        probe_client=probe,
        config=LoopConfig(max_rounds=3),
    )

    assert res_decoy.termination_reason == "certified_success"
    assert res_decoy.board.decision.answer == "SessionPool"
    assert res_decoy.final_board_drift == 0.0
    assert res_decoy.ungrounded_fields_on_board == 0
    assert res_decoy.queries_executed >= 1
    assert domain.score(res_decoy.board.model_dump(), truth_decoy) == {
        "correct": True,
        "decoy_avoided": True,
    }

    # Case 2: Under-determined (missing deterministic race repro -> honest_uncertified)
    ud_dir = tmp_path / "code_ud"
    domain.generate(str(ud_dir))
    truth_ud = domain.plant(str(ud_dir), domain.needle_menu()[1])

    res_ud = run_investigation_loop(
        domain,
        str(ud_dir),
        provider=_MockNonSQLProvider(),  # type: ignore[arg-type]
        probe_client=probe,
        config=LoopConfig(max_rounds=3),
    )
    assert res_ud.termination_reason == "honest_uncertified"
    assert res_ud.board.decision.answer == "none_verified"
    assert "deterministic_race_repro" in res_ud.blocking_fields
    assert res_ud.final_board_drift == 0.0
    assert domain.score(res_ud.board.model_dump(), truth_ud)["correct"] is True
