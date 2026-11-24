"""Non-SQL custom `ExecutionBackend` + `Domain` example for codebase concurrency auditing."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ares.contracts import (
    CostEstimate,
    ExecutionBackend,
    GroundTruth,
    GuardValidationError,
    NeedleSpec,
    Note,
    SearchQuestion,
    VerifierRegistry,
    VerifierResult,
)
from ares.engine import canonical_content_hash
from ares.store import NoteStore

MODULES: tuple[str, ...] = ("SessionPool", "TokenCache", "OrderQueue", "AuditWriter")
CODE_PIVOTAL_FIELDS: list[str] = [
    "public_entry_reachable",
    "lock_order_inversion",
    "deterministic_race_repro",
]


class ASTGraphBackend:
    """Non-SQL `ExecutionBackend` querying a JSON codebase call & lock-order graph."""

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


_CodeGraphBackend = ASTGraphBackend


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
            # High-salience 2-of-3 decoy: reachable + lock_cycle warning, but race_repro is False
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
        return ASTGraphBackend(workspace_dir)

    def pivotal_fields(self) -> list[str]:
        return list(CODE_PIVOTAL_FIELDS)

    def required_signal_fields(self) -> tuple[str, ...]:
        return tuple(CODE_PIVOTAL_FIELDS)

    def hypothesis_families(self) -> list[str]:
        return ["entry_reachability", "lock_graph_cycles", "stress_repro"]

    def search_questions(self) -> list[SearchQuestion]:
        return [
            SearchQuestion(
                question_id="buggy_module",
                question="Which module has a verified concurrency bug?",
                options=[*MODULES, "none_verified"],
                question_type="multiple_choice",
            ),
            SearchQuestion(
                question_id="sufficiently_verified",
                question="Are reachability, lock cycle, and deterministic race repro all verified?",
                options=["yes", "no"],
                question_type="yes_no",
            ),
        ]

    def resolve_answers(
        self,
        board: Any,
        questions: Sequence[SearchQuestion],
        primary_target: str | None,
    ) -> dict[str, str]:
        del board, questions
        return {"sufficiently_verified": "yes" if primary_target is not None else "no"}

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
        conc = board.get("conclusion") or board.get("decision") or {}
        ans = conc.get("answer", "unknown")
        return {
            "correct": (ans == truth.primary_target)
            if truth.is_sufficiently_verified
            else (ans == "none_verified"),
            "decoy_avoided": ans != truth.decoy_target,
        }


__all__ = [
    "ASTGraphBackend",
    "CODE_PIVOTAL_FIELDS",
    "CodebaseConcurrencyDomain",
    "MODULES",
    "_CodeGraphBackend",
]
