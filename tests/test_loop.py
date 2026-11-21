"""Fast deterministic unit tests for the 8-step investigation loop (`fusion/loop.py`)
and Part 1 calibration/grounding refinements (`fusion/patch.py`, `models/decision_probe.py`).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ares.engine import ProcessorResult, SolveEvent, TaskSpec
from ares.store import NoteStore
from contracts import (
    DecisionQuestion,
    HypothesisQuerySpec,
    InvestigationBoard,
    Note,
)
from domain.micro_bench import MicroBenchDomain
from fusion import (
    LoopConfig,
    RoundTelemetry,
    create_initial_board,
    deterministic_patch_from_notes,
    run_investigation_loop,
    validate_and_merge_patch,
)
from models import DecisionProbeClient
from models.decision_probe import _build_questions_payload
from models.gemini_provider import ModelTelemetry


# ---------------------------------------------------------------------------
# Realistic Fast Mock Provider & Mock Decision Probe
# ---------------------------------------------------------------------------


def _candidate_in_field(val: Any, service: str) -> bool:
    if isinstance(val, str):
        return val == service
    if isinstance(val, list):
        return service in val
    return False


def _infer_action_from_board(board_fields: dict[str, dict[str, Any]], service: str) -> str:
    anom_entry = board_fields.get("metric_anomaly_type", {})
    anom_val = anom_entry.get("value")
    metric_name = ""
    if isinstance(anom_val, str):
        metric_name = anom_val
    elif isinstance(anom_val, dict):
        metric_name = str(anom_val.get(service, ""))

    if metric_name == "pool_utilization_pct":
        return "scale_pool"
    if service == "inventory-db" and metric_name == "lock_wait_ms":
        return "restart"
    if service in ("payments-db", "auth-svc") and metric_name in ("lock_wait_ms", "p99_latency_ms"):
        return "rollback"
    return "rollback"


class _MockLoopProvider:
    """Fast deterministic mock of `GeminiModelProvider` over real DuckDB notes."""

    def __init__(self, *, inject_hallucinated_field: bool = False) -> None:
        self.inject_hallucinated_field = inject_hallucinated_field
        self.propose_calls = 0
        self.patch_calls = 0
        self.followup_calls = 0

    def make_propose_processor(
        self,
        *,
        schema_description: str,
        available_verifiers: list[str] | None = None,
    ) -> Any:
        del schema_description, available_verifiers

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
            hspec = HypothesisQuerySpec(
                query_sql=qsql,
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
        schema_description: str = "",
        include_images: bool = True,
        thinking_level: str = "low",
    ) -> tuple[dict[str, Any], ModelTelemetry]:
        del board, schema_description, include_images, thinking_level
        self.patch_calls += 1
        updates: dict[str, dict[str, Any]] = {}

        # Add an uncertified exploratory distractor field from ad-hoc SQL notes
        sql_notes = [n for n in new_notes if n.verifier_name is None and n.row_count > 0]
        if sql_notes:
            n0 = sql_notes[0]
            rows = n0.evidence.get("rows", [])
            if rows and isinstance(rows[0], dict):
                first_val = next(iter(rows[0].values()))
                updates["exploratory_top_caller"] = {
                    "value": first_val,
                    "sources": [n0.note_id],
                    "artifact_ids": [],
                }

        # Optionally inject an ungrounded claim on a CERTIFIED note to verify invariant guard
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
        high_voi_fields: list[str],
        schema_description: str,
        available_verifiers: list[str] | None = None,
        max_queries: int = 3,
        thinking_level: str = "low",
    ) -> tuple[list[HypothesisQuerySpec], ModelTelemetry]:
        del schema_description, available_verifiers, max_queries, thinking_level
        self.followup_calls += 1
        target = high_voi_fields[0] if high_voi_fields else "traces_error_cluster"
        spec = HypothesisQuerySpec(
            query_sql=(
                "SELECT service, status, MAX(duration_ms) AS max_dur "
                "FROM traces GROUP BY service, status ORDER BY max_dur DESC LIMIT 10"
            ),
            hypothesis_family="followup_voi_check",
            expected_signal=f"Follow-up check for {target}",
            target_fields=list(high_voi_fields),
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


class _MockDecisionBackend:
    """Deterministic `v1beta` `decisions.create` backend honoring 3-of-3 certified signal rules."""

    def __init__(self) -> None:
        self.decisions = SimpleNamespace(create=self._create)

    def _create(
        self,
        *,
        model: str,
        input: Any,
        questions: dict[str, dict[str, Any]],
    ) -> Any:
        assert model == "gemini-decision-1"
        raw_text = input[0]["text"] if isinstance(input, list) else input
        parsed = json.loads(raw_text)
        fields: dict[str, dict[str, Any]] = parsed["fields"]

        req_signals = ("logs_retry_spike", "metrics_zscore_breach", "traces_error_cluster")
        rc_options = [
            opt
            for opt in questions["root_cause"]["criteria"].keys()
            if opt != "none_verified"
        ]

        all_req_certified = all(
            fields.get(sig, {}).get("status") == "certified"
            and fields.get(sig, {}).get("value") not in ("unknown", "none", None)
            for sig in req_signals
        )

        verified_service: str | None = None
        if all_req_certified:
            for svc in rc_options:
                if all(
                    _candidate_in_field(fields[sig]["value"], svc) for sig in req_signals
                ):
                    verified_service = svc
                    break

        if verified_service is not None:
            rc_choice = verified_service
            rc_conf = 0.95
            rc_probs = {opt: 0.0 for opt in questions["root_cause"]["criteria"]}
            rc_probs[verified_service] = 0.95
            rc_probs["none_verified"] = 0.05

            act_choice = _infer_action_from_board(fields, verified_service)
            act_conf = 0.93
            act_probs = {opt: 0.02 for opt in questions["recommended_action"]["criteria"]}
            act_probs[act_choice] = 0.94
            noul_val = 0.96
        else:
            rc_choice = "none_verified"
            rc_conf = 0.93
            rc_probs = {opt: 0.0 for opt in questions["root_cause"]["criteria"]}
            rc_probs["none_verified"] = 0.93
            if rc_options:
                rc_probs[rc_options[0]] = 0.07

            act_choice = "escalate"
            act_conf = 0.94
            act_probs = {opt: 0.02 for opt in questions["recommended_action"]["criteria"]}
            act_probs["escalate"] = 0.94
            noul_val = 0.04

        answers = {
            "root_cause": SimpleNamespace(
                choice=rc_choice,
                confidence=rc_conf,
                probabilities=rc_probs,
            ),
            "recommended_action": SimpleNamespace(
                choice=act_choice,
                confidence=act_conf,
                probabilities=act_probs,
            ),
            "sufficiently_verified": SimpleNamespace(
                noul=noul_val,
                confidence=max(noul_val, 1.0 - noul_val),
            ),
        }
        usage = SimpleNamespace(
            total_input_tokens=110,
            total_output_tokens=0,
            total_thought_tokens=0,
            total_tokens=110,
        )
        return SimpleNamespace(answers=answers, usage=usage)


# ---------------------------------------------------------------------------
# Part 1 Verification Tests: `metric_anomaly_type` & Calibration Rubrics
# ---------------------------------------------------------------------------


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
    patch_single = deterministic_patch_from_notes([n_single])
    assert patch_single.updates["metric_anomaly_type"].value == "lock_wait_ms"
    assert patch_single.updates["metric_anomaly_type"].sources == ["n_m1"]

    board = create_initial_board(MicroBenchDomain().decision_questions())
    merged_single, rep_single = validate_and_merge_patch(board, patch_single, store)
    assert rep_single.rejected_fields == {}
    assert merged_single.fields["metric_anomaly_type"].status == "certified"
    assert merged_single.fields["metric_anomaly_type"].value == "lock_wait_ms"

    # Multi-candidate decoy metric note
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
    patch_multi = deterministic_patch_from_notes([n_multi])
    assert patch_multi.updates["metric_anomaly_type"].value == {
        "cache-redis": "p99_latency_ms",
        "payments-db": "pool_utilization_pct",
    }
    merged_multi, rep_multi = validate_and_merge_patch(board, patch_multi, store_multi)
    assert rep_multi.rejected_fields == {}
    assert merged_multi.fields["metric_anomaly_type"].status == "certified"


def test_decision_probe_rubrics_handle_signal_lists_and_exploratory_fields() -> None:
    questions: list[DecisionQuestion] = MicroBenchDomain().decision_questions()
    payload = _build_questions_payload(questions)

    sv_inst = payload["sufficiently_verified"]["instructions"]
    assert "logs_retry_spike" in sv_inst
    assert "metrics_zscore_breach" in sv_inst
    assert "traces_error_cluster" in sv_inst
    assert "noul >= 0.5" in sv_inst

    rc_spec = payload["root_cause"]
    assert "list of candidate services" in rc_spec["instructions"]
    assert "logs_retry_spike" in rc_spec["criteria"]["none_verified"]
    assert "ALL THREE required signal fields" in rc_spec["criteria"]["payments-db"]


# ---------------------------------------------------------------------------
# Part 2 & 3 End-to-End Loop Tests over Real MicroBenchDomain Parquet Shards
# ---------------------------------------------------------------------------


def test_loop_multi_clue_join_certified_success_and_artifacts(tmp_path: Path) -> None:
    domain = MicroBenchDomain()
    case_dir = tmp_path / "mc_01"
    domain.generate(str(case_dir), size_gb=0.002, seed=42)
    needle = next(n for n in domain.needle_menu() if n.case_id == "mc_01_payments_db_lock")
    truth = domain.plant(str(case_dir), needle)

    provider = _MockLoopProvider(inject_hallucinated_field=True)
    probe = DecisionProbeClient(client=_MockDecisionBackend())
    round_callbacks: list[RoundTelemetry] = []
    ares_events: list[SolveEvent] = []

    res = run_investigation_loop(
        domain,
        str(case_dir),
        provider=provider,  # type: ignore[arg-type]
        probe_client=probe,
        config=LoopConfig(max_rounds=3),
        on_round=lambda rt, _b: round_callbacks.append(rt),
        on_ares_event=ares_events.append,
    )

    assert res.termination_reason == "certified_success"
    assert res.blocking_fields == []
    assert len(res.rounds) == 1
    assert len(round_callbacks) == 1
    assert len(ares_events) > 0

    # Hallucinated Carbon field was deterministically caught & rejected; final board has 0 violations
    assert res.invariant_violations_caught == 1
    assert res.ungrounded_fields_on_board == 0
    assert res.illegal_cert_upgrades_on_board == 0
    assert res.final_board_drift == 0.0

    # Multimodal PNG artifact generated by verify_three_way_root_cause
    assert len(res.artifacts) >= 1
    assert Path(res.artifacts[0].path).is_file()
    assert res.board.fields["metric_anomaly_type"].value == "lock_wait_ms"
    assert res.board.fields["metric_anomaly_type"].status == "certified"

    scored = domain.score(res.board.model_dump(), truth)
    assert scored["overall_pass"] is True
    assert scored["predicted_root_cause"] == "payments-db"
    assert scored["predicted_action"] == "rollback"
    assert scored["false_certainty"] is False
    assert scored["pivotal_influence_mean"] > scored["distractor_influence_mean"]


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
    probe = DecisionProbeClient(client=_MockDecisionBackend())

    res = run_investigation_loop(
        domain,
        str(case_dir),
        provider=provider,  # type: ignore[arg-type]
        probe_client=probe,
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
    assert scored["pivotal_influence_mean"] is not None
    assert scored["distractor_influence_mean"] is not None
    assert scored["pivotal_influence_mean"] > scored["distractor_influence_mean"]


def test_loop_under_determined_terminates_honest_uncertified_without_false_certainty(
    tmp_path: Path,
) -> None:
    domain = MicroBenchDomain()
    case_dir = tmp_path / "ud_01"
    domain.generate(str(case_dir), size_gb=0.002, seed=42)
    needle = next(n for n in domain.needle_menu() if n.case_id == "ud_01_missing_traces")
    truth = domain.plant(str(case_dir), needle)

    provider = _MockLoopProvider()
    probe = DecisionProbeClient(client=_MockDecisionBackend())

    res = run_investigation_loop(
        domain,
        str(case_dir),
        provider=provider,  # type: ignore[arg-type]
        probe_client=probe,
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


def test_build_baseline_board_for_scoring_fairly_credits_and_penalizes(
    tmp_path: Path,
) -> None:
    from benchmark import build_baseline_board_for_scoring

    domain = MicroBenchDomain()
    needles = domain.needle_menu()
    mc_needle = next(n for n in needles if n.category == "multi_clue_join")
    ud_needle = next(n for n in needles if n.category == "under_determined")

    # Confident & verified right answer on multi-clue needle -> overall_pass True
    b_board_ok = build_baseline_board_for_scoring(
        {
            "root_cause": mc_needle.service,
            "recommended_action": mc_needle.recommended_action,
            "sufficiently_verified": "yes",
            "confidence": 0.91,
        }
    )
    mc_truth = domain.plant(str(tmp_path / "mc_score"), mc_needle)
    mc_score = domain.score(b_board_ok, mc_truth)
    assert mc_score["board_certified"] is True
    assert mc_score["overall_pass"] is True

    # Overconfident 'yes' on under-determined needle -> false_certainty True!
    b_board_fc = build_baseline_board_for_scoring(
        {
            "root_cause": ud_needle.service,
            "recommended_action": "rollback",
            "sufficiently_verified": "yes",
            "confidence": 0.90,
        }
    )
    ud_truth = domain.plant(str(tmp_path / "ud_score"), ud_needle)
    ud_score = domain.score(b_board_fc, ud_truth)
    assert ud_score["false_certainty"] is True
    assert ud_score["overall_pass"] is False

