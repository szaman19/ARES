"""Deterministic offline tests for the ARES orchestration & context loop (`ares/loop.py`)."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ares.context import (
    create_initial_board,
    deterministic_patch_from_notes,
    validate_and_merge_patch,
)
from ares.contracts import (
    ExecutionBackend,
    HypothesisSpec,
    InvestigationBoard,
    Note,
    SearchQuestion,
    VerifierRegistry,
    VerifierResult,
)
from ares.engine import ProcessorResult, SolveEvent, TaskSpec
from ares.loop import LoopConfig, RoundTelemetry, run_investigation_loop
from ares.models import ContextLengthExceededError, ModelTelemetry
from ares.store import NoteStore
from examples.incident_rca.benchmark import build_baseline_board_for_scoring
from examples.incident_rca.domain import (
    INCIDENT_PIVOTAL_FIELDS,
    MicroBenchDomain,
    extract_incident_note_fields,
)


class _MockLoopProvider:
    """Fast deterministic mock of `GeminiModelProvider` over real DuckDB notes."""

    def __init__(
        self,
        *,
        inject_hallucinated_field: bool = False,
        inject_extra_distractors: bool = False,
    ) -> None:
        self.inject_hallucinated_field = inject_hallucinated_field
        self.inject_extra_distractors = inject_extra_distractors
        self.propose_calls = 0
        self.patch_calls = 0
        self.followup_calls = 0

    def make_propose_processor(
        self,
        *,
        environment_description: str,
        magnet_rules: str = "",
        pivotal_fields: Sequence[str] = (),
        available_verifiers: Sequence[str] | None = None,
    ) -> Any:
        del environment_description, magnet_rules, pivotal_fields, available_verifiers

        def _proc(spec: TaskSpec, dep_outputs: dict[str, Any]) -> ProcessorResult:
            del dep_outputs
            self.propose_calls += 1
            fam = str((spec.context or {}).get("hypothesis_family", "general"))
            sql_by_family = {
                "upstream_retry_spike": (
                    "SELECT downstream_service, COUNT(*) AS c "
                    "FROM service_logs WHERE level = 'ERROR' AND retry_count >= 3 "
                    "GROUP BY downstream_service ORDER BY c DESC LIMIT 10"
                ),
                "metric_baseline_zscore_breach": (
                    "SELECT service, metric_name, MAX((value - baseline_mean) / baseline_std) AS max_z "
                    "FROM metrics GROUP BY service, metric_name ORDER BY max_z DESC LIMIT 10"
                ),
                "downstream_trace_timeout_cluster": (
                    "SELECT service, COUNT(*) AS err_spans "
                    "FROM traces WHERE status = 'ERROR' AND duration_ms >= 800 "
                    "GROUP BY service ORDER BY err_spans DESC LIMIT 10"
                ),
                "cross_service_three_way_join": (
                    "SELECT l.downstream_service, COUNT(DISTINCT l.trace_id) AS n "
                    "FROM service_logs l JOIN traces t ON l.trace_id = t.trace_id "
                    "WHERE l.level = 'ERROR' AND t.status = 'ERROR' "
                    "GROUP BY l.downstream_service ORDER BY n DESC LIMIT 10"
                ),
            }
            qsql = sql_by_family.get(
                fam,
                "SELECT service, COUNT(*) AS c FROM service_logs GROUP BY service LIMIT 5",
            )
            hspec = HypothesisSpec(
                query=qsql,
                hypothesis_family=fam,
                expected_signal=f"Test {fam}",
                requested_under=spec.requested_under,
            )
            return ProcessorResult(
                output=[hspec],
                prompt_tokens=100,
                completion_tokens=40,
                model_calls=1,
            )

        return _proc

    def propose_board_patch(
        self,
        *,
        board: InvestigationBoard,
        new_notes: list[Note],
        environment_description: str = "",
        pivotal_fields: Sequence[str] = (),
        include_images: bool = True,
        thinking_level: str = "low",
    ) -> tuple[dict[str, Any], ModelTelemetry]:
        del board, environment_description, pivotal_fields, include_images, thinking_level
        self.patch_calls += 1
        updates: dict[str, dict[str, Any]] = {}

        query_notes = [n for n in new_notes if n.verifier_name is None and n.row_count > 0]
        if query_notes:
            n0 = query_notes[0]
            rows = n0.evidence.get("rows", [])
            if rows and isinstance(rows[0], dict):
                first_val = next(iter(rows[0].values()))
                updates["exploratory_top_caller"] = {
                    "value": first_val,
                    "sources": [n0.note_id],
                    "artifact_ids": [],
                }
            if self.inject_extra_distractors:
                for idx, qn in enumerate(query_notes):
                    updates[f"distractor_probe_{self.patch_calls}_{idx}"] = {
                        "value": qn.summary,
                        "sources": [qn.note_id],
                        "artifact_ids": [],
                    }

        cert_notes = [n for n in new_notes if n.certification == "CERTIFIED"]
        if self.inject_hallucinated_field and cert_notes:
            updates["hallucinated_claim"] = {
                "value": "completely-ungrounded-service-xyz",
                "sources": [cert_notes[0].note_id],
                "artifact_ids": [],
            }

        return (
            {"updates": updates, "rationale": "Mock Carbon patch synthesis."},
            ModelTelemetry(
                prompt_tokens=200,
                completion_tokens=60,
                thought_tokens=20,
                total_tokens=280,
                model_calls=1,
                latency_ms=5.0,
            ),
        )

    def propose_followup_tasks(
        self,
        *,
        board: InvestigationBoard,
        target_gap_fields: list[str],
        environment_description: str,
        magnet_rules: str = "",
        available_verifiers: Sequence[str] | None = None,
        max_queries: int = 3,
        thinking_level: str = "low",
    ) -> tuple[list[HypothesisSpec], ModelTelemetry]:
        del environment_description, magnet_rules, available_verifiers, max_queries, thinking_level
        self.followup_calls += 1
        target = target_gap_fields[0] if target_gap_fields else "traces_error_cluster"
        spec = HypothesisSpec(
            query=(
                "SELECT service, status, MAX(duration_ms) AS max_dur "
                "FROM traces GROUP BY service, status ORDER BY max_dur DESC LIMIT 10"
            ),
            hypothesis_family="followup_gap_check",
            expected_signal=f"Follow-up check for {target}",
            target_fields=list(target_gap_fields),
            requested_under=board.board_version,
        )
        return (
            [spec],
            ModelTelemetry(
                prompt_tokens=140,
                completion_tokens=35,
                total_tokens=175,
                model_calls=1,
                latency_ms=4.0,
            ),
        )


def test_deterministic_patch_populates_metric_anomaly_type_single_and_multi() -> None:
    store = NoteStore()
    n_single = store.append(
        Note(
            note_id="n_m1",
            certification="CERTIFIED",
            verifier_name="verify_metric_zscore_breach",
            row_count=1,
            evidence={
                "candidates": [
                    {
                        "service": "payments-db",
                        "shard_id": "shard-01",
                        "metric_name": "lock_wait_ms",
                        "breach_buckets": 6,
                        "max_zscore": 6.4,
                    }
                ]
            },
        )
    )
    patch_single = deterministic_patch_from_notes(
        [n_single], note_extractor=extract_incident_note_fields
    )
    assert patch_single.updates["metric_anomaly_type"].value == "lock_wait_ms"
    assert patch_single.updates["metric_anomaly_type"].sources == ["n_m1"]

    board = create_initial_board(
        MicroBenchDomain().search_questions(), seed_fields=INCIDENT_PIVOTAL_FIELDS
    )
    merged_single, rep_single = validate_and_merge_patch(
        board,
        patch_single,
        store,
        seed_fields=INCIDENT_PIVOTAL_FIELDS,
        protected_fields=INCIDENT_PIVOTAL_FIELDS,
        note_extractor=extract_incident_note_fields,
    )
    assert rep_single.rejected_fields == {}
    assert merged_single.fields["metric_anomaly_type"].status == "certified"
    assert merged_single.fields["metric_anomaly_type"].value == "lock_wait_ms"

    store_multi = NoteStore()
    n_multi = store_multi.append(
        Note(
            note_id="n_m2",
            certification="CERTIFIED",
            verifier_name="verify_metric_zscore_breach",
            row_count=2,
            evidence={
                "candidates": [
                    {
                        "service": "cache-redis",
                        "shard_id": "shard-02",
                        "metric_name": "p99_latency_ms",
                        "breach_buckets": 8,
                        "max_zscore": 7.1,
                    },
                    {
                        "service": "payments-db",
                        "shard_id": "shard-01",
                        "metric_name": "pool_utilization_pct",
                        "breach_buckets": 6,
                        "max_zscore": 6.2,
                    },
                ]
            },
        )
    )
    patch_multi = deterministic_patch_from_notes(
        [n_multi], note_extractor=extract_incident_note_fields
    )
    assert patch_multi.updates["metric_anomaly_type"].value == {
        "cache-redis": "p99_latency_ms",
        "payments-db": "pool_utilization_pct",
    }
    merged_multi, rep_multi = validate_and_merge_patch(
        board,
        patch_multi,
        store_multi,
        seed_fields=INCIDENT_PIVOTAL_FIELDS,
        protected_fields=INCIDENT_PIVOTAL_FIELDS,
        note_extractor=extract_incident_note_fields,
    )
    assert rep_multi.rejected_fields == {}
    assert merged_multi.fields["metric_anomaly_type"].status == "certified"


def test_loop_multi_clue_join_certified_success_and_artifacts(tmp_path: Path) -> None:
    domain = MicroBenchDomain()
    case_dir = tmp_path / "mc_01"
    domain.generate(str(case_dir), size_gb=0.002, seed=42)
    needle = next(n for n in domain.needle_menu() if n.case_id == "mc_01_payments_db_lock")
    truth = domain.plant(str(case_dir), needle)

    provider = _MockLoopProvider(inject_hallucinated_field=True)
    round_callbacks: list[RoundTelemetry] = []
    ares_events: list[SolveEvent] = []

    res = run_investigation_loop(
        domain,
        str(case_dir),
        provider=provider,  # type: ignore[arg-type]
        config=LoopConfig(max_rounds=3),
        on_round=lambda rt, _b: round_callbacks.append(rt),
        on_ares_event=ares_events.append,
    )

    assert res.termination_reason == "certified_success"
    assert res.blocking_fields == []
    assert len(res.rounds) == 1
    assert len(round_callbacks) == 1
    assert len(ares_events) > 0
    assert res.evaluator_calls == 1
    assert res.rounds[0].conclusion_answer == "payments-db"
    assert res.rounds[0].conclusion_confidence == 1.0

    assert res.invariant_violations_caught == 1
    assert res.ungrounded_fields_on_board == 0
    assert res.illegal_cert_upgrades_on_board == 0
    assert res.final_board_drift == 0.0

    assert len(res.artifacts) >= 1
    assert Path(res.artifacts[0].path).is_file()
    assert res.board.fields["metric_anomaly_type"].value == "lock_wait_ms"
    assert res.board.fields["metric_anomaly_type"].status == "certified"

    scored = domain.score(res.board.model_dump(), truth)
    assert scored["overall_pass"] is True
    assert scored["predicted_root_cause"] == "payments-db"
    assert scored["predicted_action"] == "rollback"
    assert scored["false_certainty"] is False
    assert scored["pivotal_priority_mean"] > scored["distractor_priority_mean"]


def test_loop_look_alike_decoy_avoids_decoy_and_ranks_pivotal_above_distractor(
    tmp_path: Path,
) -> None:
    domain = MicroBenchDomain()
    case_dir = tmp_path / "decoy_01"
    domain.generate(str(case_dir), size_gb=0.002, seed=42)
    needle = next(
        n for n in domain.needle_menu() if n.case_id == "decoy_01_payments_vs_cache"
    )
    truth = domain.plant(str(case_dir), needle)

    provider = _MockLoopProvider()
    res = run_investigation_loop(
        domain,
        str(case_dir),
        provider=provider,  # type: ignore[arg-type]
        config=LoopConfig(max_rounds=3),
    )

    assert res.termination_reason == "certified_success"
    assert res.ungrounded_fields_on_board == 0
    assert res.illegal_cert_upgrades_on_board == 0
    assert res.final_board_drift == 0.0
    assert isinstance(res.board.fields["metric_anomaly_type"].value, dict)
    assert (
        res.board.fields["metric_anomaly_type"].value.get("payments-db")
        == "pool_utilization_pct"
    )

    scored = domain.score(res.board.model_dump(), truth)
    assert scored["overall_pass"] is True
    assert scored["decoy_avoided"] is True
    assert scored["decoy_fallen_for"] is False
    assert scored["predicted_root_cause"] == "payments-db"
    assert scored["predicted_action"] == "scale_pool"
    assert scored["pivotal_priority_mean"] is not None
    assert scored["distractor_priority_mean"] is not None
    assert scored["pivotal_priority_mean"] > scored["distractor_priority_mean"]


def test_loop_under_determined_terminates_honest_uncertified_without_false_certainty(
    tmp_path: Path,
) -> None:
    domain = MicroBenchDomain()
    case_dir = tmp_path / "ud_01"
    domain.generate(str(case_dir), size_gb=0.002, seed=42)
    needle = next(n for n in domain.needle_menu() if n.case_id == "ud_01_missing_traces")
    truth = domain.plant(str(case_dir), needle)

    provider = _MockLoopProvider()
    res = run_investigation_loop(
        domain,
        str(case_dir),
        provider=provider,  # type: ignore[arg-type]
        config=LoopConfig(max_rounds=3),
    )

    assert res.termination_reason == "honest_uncertified"
    assert len(res.rounds) == 2
    assert provider.followup_calls == 1
    assert "traces_error_cluster" in res.blocking_fields
    assert res.ungrounded_fields_on_board == 0
    assert res.illegal_cert_upgrades_on_board == 0
    assert res.final_board_drift == 0.0

    scored = domain.score(res.board.model_dump(), truth)
    assert scored["overall_pass"] is True
    assert scored["false_certainty"] is False
    assert scored["board_certified"] is False
    assert scored["predicted_root_cause"] == "none_verified"
    assert scored["predicted_action"] == "escalate"


def test_loop_automatic_context_archiving_enforces_max_active_fields(
    tmp_path: Path,
) -> None:
    domain = MicroBenchDomain()
    case_dir = tmp_path / "ud_archive"
    domain.generate(str(case_dir), size_gb=0.002, seed=42)
    needle = next(n for n in domain.needle_menu() if n.case_id == "ud_01_missing_traces")
    domain.plant(str(case_dir), needle)

    provider = _MockLoopProvider(inject_extra_distractors=True)
    res = run_investigation_loop(
        domain,
        str(case_dir),
        provider=provider,  # type: ignore[arg-type]
        config=LoopConfig(max_rounds=3, max_active_fields=5),
    )

    assert res.termination_reason == "honest_uncertified"
    assert len(res.board.fields) <= 5
    assert len(res.board.archived_fields) >= 1
    for piv in INCIDENT_PIVOTAL_FIELDS:
        assert piv in res.board.fields
    assert res.final_board_drift == 0.0
    assert res.ungrounded_fields_on_board == 0


def test_build_baseline_board_for_scoring_fairly_credits_and_penalizes(
    tmp_path: Path,
) -> None:
    domain = MicroBenchDomain()
    needles = domain.needle_menu()
    mc_needle = next(n for n in needles if n.category == "multi_clue_join")
    ud_needle = next(n for n in needles if n.category == "under_determined")

    b_board_ok = build_baseline_board_for_scoring(
        {
            "answers": {
                "root_cause": mc_needle.target_entity,
                "recommended_action": mc_needle.expected_decisions["recommended_action"],
                "sufficiently_verified": "yes",
            },
            "confidence": 0.91,
        },
        domain.search_questions(),
    )
    mc_truth = domain.plant(str(tmp_path / "mc_score"), mc_needle)
    mc_score = domain.score(b_board_ok, mc_truth)
    assert mc_score["board_certified"] is True
    assert mc_score["overall_pass"] is True

    b_board_fc = build_baseline_board_for_scoring(
        {
            "answers": {
                "root_cause": ud_needle.target_entity,
                "recommended_action": "rollback",
                "sufficiently_verified": "yes",
            },
            "confidence": 0.90,
        },
        domain.search_questions(),
    )
    ud_truth = domain.plant(str(tmp_path / "ud_score"), ud_needle)
    ud_score = domain.score(b_board_fc, ud_truth)
    assert ud_score["false_certainty"] is True
    assert ud_score["overall_pass"] is False


def test_loop_lossless_note_batching_and_peak_prompt_tokens(tmp_path: Path) -> None:
    domain = MicroBenchDomain()
    case_dir = tmp_path / "mc_batched"
    domain.generate(str(case_dir), size_gb=0.002, seed=42)
    needle = next(n for n in domain.needle_menu() if n.case_id == "mc_01_payments_db_lock")
    truth = domain.plant(str(case_dir), needle)

    provider = _MockLoopProvider()
    res = run_investigation_loop(
        domain,
        str(case_dir),
        provider=provider,  # type: ignore[arg-type]
        config=LoopConfig(max_rounds=3, max_notes_per_patch=2),
    )

    assert res.termination_reason == "certified_success"
    assert res.rounds[0].patch_batches >= 2
    assert res.rounds[0].peak_prompt_tokens == 200
    assert res.peak_prompt_tokens == 200
    assert res.final_board_drift == 0.0
    assert res.ungrounded_fields_on_board == 0
    scored = domain.score(res.board.model_dump(), truth)
    assert scored["overall_pass"] is True


def test_loop_parallel_followup_and_attempted_queries_memory(tmp_path: Path) -> None:
    domain = MicroBenchDomain()
    case_dir = tmp_path / "ud_parallel_followup"
    domain.generate(str(case_dir), size_gb=0.002, seed=42)
    needle = next(n for n in domain.needle_menu() if n.case_id == "ud_01_missing_traces")
    domain.plant(str(case_dir), needle)

    class _ParallelFollowupProvider(_MockLoopProvider):
        def __init__(self) -> None:
            super().__init__()
            self._lock = threading.Lock()
            self.followup_records: list[tuple[list[str], list[str]]] = []

        def propose_followup_tasks(
            self,
            *,
            board: InvestigationBoard,
            target_gap_fields: list[str],
            environment_description: str,
            magnet_rules: str = "",
            available_verifiers: Sequence[str] | None = None,
            max_queries: int = 3,
            thinking_level: str = "low",
            attempted_queries: Sequence[str] = (),
        ) -> tuple[list[HypothesisSpec], ModelTelemetry]:
            del environment_description, magnet_rules, available_verifiers, max_queries, thinking_level
            with self._lock:
                self.followup_calls += 1
                self.followup_records.append((list(target_gap_fields), list(attempted_queries)))
            target = target_gap_fields[0] if target_gap_fields else "gap"
            spec = HypothesisSpec(
                query=f"SELECT COUNT(*) AS cnt_{target} FROM traces WHERE status = 'ERROR'",
                hypothesis_family=f"followup_{target}",
                expected_signal=f"Resolve {target}",
                target_fields=list(target_gap_fields),
                requested_under=board.board_version,
            )
            return (
                [spec],
                ModelTelemetry(
                    prompt_tokens=165,
                    completion_tokens=30,
                    peak_prompt_tokens=165,
                    model_calls=1,
                ),
            )

    provider = _ParallelFollowupProvider()
    res = run_investigation_loop(
        domain,
        str(case_dir),
        provider=provider,  # type: ignore[arg-type]
        config=LoopConfig(max_rounds=2, max_concurrent_workers=4, parallel_followup=True),
    )

    assert len(res.rounds) == 2
    assert len(provider.followup_records) >= 2
    for gap_fields, attempted in provider.followup_records:
        assert len(gap_fields) == 1
        assert len(attempted) >= 1
    assert res.rounds[1].proposed_hypotheses >= 2
    assert res.final_board_drift == 0.0


def test_loop_stagnant_rounds_stops_honestly_with_verifier_exhaustion_disabled(
    tmp_path: Path,
) -> None:
    domain = MicroBenchDomain()
    case_dir = tmp_path / "ud_stagnant"
    domain.generate(str(case_dir), size_gb=0.002, seed=42)
    needle = next(n for n in domain.needle_menu() if n.case_id == "ud_01_missing_traces")
    domain.plant(str(case_dir), needle)

    class _StagnantFollowupProvider(_MockLoopProvider):
        def propose_board_patch(
            self,
            *,
            board: InvestigationBoard,
            new_notes: list[Note],
            environment_description: str = "",
            pivotal_fields: Sequence[str] = (),
            include_images: bool = True,
            thinking_level: str = "low",
        ) -> tuple[dict[str, Any], ModelTelemetry]:
            if board.board_version >= 1:
                return (
                    {"updates": {}, "rationale": "No new signal change."},
                    ModelTelemetry(prompt_tokens=180, completion_tokens=20, model_calls=1),
                )
            return super().propose_board_patch(
                board=board,
                new_notes=new_notes,
                environment_description=environment_description,
                pivotal_fields=pivotal_fields,
                include_images=include_images,
                thinking_level=thinking_level,
            )

        def propose_followup_tasks(
            self,
            *,
            board: InvestigationBoard,
            target_gap_fields: list[str],
            environment_description: str,
            magnet_rules: str = "",
            available_verifiers: Sequence[str] | None = None,
            max_queries: int = 3,
            thinking_level: str = "low",
            attempted_queries: Sequence[str] = (),
        ) -> tuple[list[HypothesisSpec], ModelTelemetry]:
            del environment_description, magnet_rules, available_verifiers, max_queries, thinking_level, attempted_queries
            self.followup_calls += 1
            spec = HypothesisSpec(
                query=f"SELECT COUNT(*) AS n FROM service_logs WHERE retry_count > {1000 + self.followup_calls}",
                hypothesis_family="followup_probe",
                expected_signal="Novel non-mutating check",
                target_fields=list(target_gap_fields),
                requested_under=board.board_version,
            )
            return [spec], ModelTelemetry(prompt_tokens=130, completion_tokens=25, model_calls=1)

    provider = _StagnantFollowupProvider()
    res = run_investigation_loop(
        domain,
        str(case_dir),
        provider=provider,  # type: ignore[arg-type]
        config=LoopConfig(
            max_rounds=6,
            stop_when_all_verifiers_exhausted=False,
            max_stagnant_rounds=2,
        ),
    )

    assert res.termination_reason == "honest_uncertified"
    # Round 1 mutates board (stagnant=0), Rounds 2 & 3 make zero field progress (stagnant=2 -> stop)
    assert len(res.rounds) == 3
    assert res.final_board_drift == 0.0


def test_loop_checkpoint_save_and_resume_round_trip(tmp_path: Path) -> None:
    class _TwoStageDomain:
        name = "two_stage_checkpoint_domain"

        def generate(self, out_dir: str, size_gb: float = 0.0, seed: int = 1) -> None:
            del out_dir, size_gb, seed

        def plant(self, out_dir: str, needle: Any) -> Any:
            del out_dir, needle
            return None

        def needle_menu(self) -> list[Any]:
            return []

        def environment_description(self) -> str:
            return "Two-stage verification environment."

        def magnet_rules(self) -> str:
            return "Bind `verify_stage2` with stage=2 on follow-up."

        def create_backend(self, workspace_dir: str) -> ExecutionBackend | None:
            del workspace_dir
            return None

        def register_verifiers(self, registry: VerifierRegistry) -> None:
            def _v_stage2(params: dict[str, Any]) -> VerifierResult:
                if int(params.get("stage", 1)) >= 2:
                    return VerifierResult(
                        verifier_name="verify_stage2",
                        passed=True,
                        certification="CERTIFIED",
                        evidence={"candidates": ["ModuleAlpha"]},
                        row_count=1,
                        summary="Stage 2 certified ModuleAlpha",
                    )
                return VerifierResult(
                    verifier_name="verify_stage2",
                    passed=False,
                    certification="UNCERTIFIED",
                    evidence={"candidates": []},
                    row_count=0,
                    summary="Stage 2 requires follow-up trigger",
                )

            registry.register("verify_stage2", _v_stage2, trusted=True)

        def search_questions(self) -> list[SearchQuestion]:
            return [
                SearchQuestion(
                    question_id="target_module",
                    question="Which module is verified?",
                    options=["ModuleAlpha", "none_verified"],
                ),
                SearchQuestion(
                    question_id="sufficiently_verified",
                    question="Is stage2 verified?",
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
            return {"sufficiently_verified": "yes" if primary_target else "no"}

        def hypothesis_families(self) -> list[str]:
            return ["stage1_probe"]

        def pivotal_fields(self) -> list[str]:
            return ["stage2_signal"]

        def required_signal_fields(self) -> tuple[str, ...]:
            return ("stage2_signal",)

        def extract_field_patches_from_note(self, note: Note) -> dict[str, Any] | None:
            if note.verifier_name != "verify_stage2":
                return None
            cands = note.evidence.get("candidates", [])
            return {"stage2_signal": cands[0] if cands else "none"}

        def score(self, board: dict[str, Any], truth: Any) -> dict[str, Any]:
            del board, truth
            return {}

    class _TwoStageProvider(_MockLoopProvider):
        def make_propose_processor(
            self,
            *,
            environment_description: str,
            magnet_rules: str = "",
            pivotal_fields: Sequence[str] = (),
            available_verifiers: Sequence[str] | None = None,
        ) -> Any:
            del environment_description, magnet_rules, pivotal_fields, available_verifiers

            def _proc(spec: TaskSpec, dep_outputs: dict[str, Any]) -> ProcessorResult:
                del dep_outputs
                self.propose_calls += 1
                return ProcessorResult(
                    output=[
                        HypothesisSpec(
                            query="",
                            params={"stage": 1},
                            hypothesis_family="stage1_probe",
                            expected_signal="initial stage1 probe",
                            verifier_name="verify_stage2",
                            requested_under=spec.requested_under,
                        )
                    ],
                    prompt_tokens=90,
                    completion_tokens=20,
                    model_calls=1,
                )

            return _proc

        def propose_board_patch(
            self,
            *,
            board: InvestigationBoard,
            new_notes: list[Note],
            environment_description: str = "",
            pivotal_fields: Sequence[str] = (),
            include_images: bool = True,
            thinking_level: str = "low",
        ) -> tuple[dict[str, Any], ModelTelemetry]:
            del board, new_notes, environment_description, pivotal_fields, include_images, thinking_level
            self.patch_calls += 1
            return {"updates": {}, "rationale": "Canonical stage merge."}, ModelTelemetry(
                prompt_tokens=110, completion_tokens=15, model_calls=1
            )

        def propose_followup_tasks(
            self,
            *,
            board: InvestigationBoard,
            target_gap_fields: list[str],
            environment_description: str,
            magnet_rules: str = "",
            available_verifiers: Sequence[str] | None = None,
            max_queries: int = 3,
            thinking_level: str = "low",
            attempted_queries: Sequence[str] = (),
        ) -> tuple[list[HypothesisSpec], ModelTelemetry]:
            del target_gap_fields, environment_description, magnet_rules, available_verifiers, max_queries, thinking_level, attempted_queries
            self.followup_calls += 1
            return (
                [
                    HypothesisSpec(
                        query="",
                        params={"stage": 2},
                        hypothesis_family="stage2_followup",
                        expected_signal="trigger stage2 certification",
                        verifier_name="verify_stage2",
                        requested_under=board.board_version,
                    )
                ],
                ModelTelemetry(prompt_tokens=120, completion_tokens=25, model_calls=1),
            )

    domain = _TwoStageDomain()
    ckpt_file = tmp_path / "checkpoints" / "ares_ckpt.json"

    res_part1 = run_investigation_loop(
        domain,
        str(tmp_path),
        provider=_TwoStageProvider(),  # type: ignore[arg-type]
        config=LoopConfig(max_rounds=1, checkpoint_path=str(ckpt_file)),
    )
    assert res_part1.termination_reason == "max_rounds"
    assert len(res_part1.rounds) == 1
    assert ckpt_file.is_file()

    # Resume with a fresh provider instance from Round 1 checkpoint through Round 3
    res_resumed = run_investigation_loop(
        domain,
        str(tmp_path),
        provider=_TwoStageProvider(),  # type: ignore[arg-type]
        config=LoopConfig(
            max_rounds=3,
            resume_from_checkpoint=str(ckpt_file),
            checkpoint_path=str(ckpt_file),
        ),
    )
    assert res_resumed.termination_reason == "certified_success"
    assert res_resumed.board.conclusion.answer == "ModuleAlpha"
    assert [r.round_index for r in res_resumed.rounds] == [1, 2]
    assert [n.note_id for n in res_resumed.notes] == ["n_0001", "n_0002"]
    assert res_resumed.final_board_drift == 0.0
    assert res_resumed.ungrounded_fields_on_board == 0


def test_loop_context_length_exceeded_terminates_cleanly(tmp_path: Path) -> None:
    domain = MicroBenchDomain()
    case_dir = tmp_path / "ctx_overflow"
    domain.generate(str(case_dir), size_gb=0.002, seed=42)
    needle = next(n for n in domain.needle_menu() if n.case_id == "mc_01_payments_db_lock")
    domain.plant(str(case_dir), needle)

    class _ProposeOverflowProvider(_MockLoopProvider):
        def make_propose_processor(
            self,
            *,
            environment_description: str,
            magnet_rules: str = "",
            pivotal_fields: Sequence[str] = (),
            available_verifiers: Sequence[str] | None = None,
        ) -> Any:
            del environment_description, magnet_rules, pivotal_fields, available_verifiers

            def _proc(spec: TaskSpec, dep_outputs: dict[str, Any]) -> ProcessorResult:
                del spec, dep_outputs
                raise ContextLengthExceededError("Prompt exceeded 4096 context tokens.")

            return _proc

    res_prop = run_investigation_loop(
        domain,
        str(case_dir),
        provider=_ProposeOverflowProvider(),  # type: ignore[arg-type]
        config=LoopConfig(max_rounds=3),
    )
    assert res_prop.termination_reason == "context_overflow"
    assert res_prop.final_board_drift == 0.0

    class _PatchOverflowProvider(_MockLoopProvider):
        def propose_board_patch(
            self,
            *,
            board: InvestigationBoard,
            new_notes: list[Note],
            environment_description: str = "",
            pivotal_fields: Sequence[str] = (),
            include_images: bool = True,
            thinking_level: str = "low",
        ) -> tuple[dict[str, Any], ModelTelemetry]:
            del board, new_notes, environment_description, pivotal_fields, include_images, thinking_level
            raise ContextLengthExceededError("Synthesis prompt exceeded context limit.")

    res_patch = run_investigation_loop(
        domain,
        str(case_dir),
        provider=_PatchOverflowProvider(),  # type: ignore[arg-type]
        config=LoopConfig(max_rounds=3),
    )
    assert res_patch.termination_reason == "context_overflow"
    assert res_patch.final_board_drift == 0.0
