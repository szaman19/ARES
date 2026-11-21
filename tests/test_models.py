"""Unit and live smoke tests for `models/gemini_provider.py` and `models/decision_probe.py`."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dotenv import load_dotenv

from ares.engine import TaskSpec, solve
from contracts import (
    ArtifactRef,
    BoardDecision,
    BoardField,
    DecisionQuestion,
    HypothesisQuerySpec,
    InvestigationBoard,
    Note,
)
from domain.micro_bench import MicroBenchDomain
from models import (
    DecisionProbeClient,
    GeminiModelProvider,
    ablate_board_fields,
    perturb_board_field,
    serialize_board_for_decision,
)

# Minimal valid 1x1 transparent PNG bytes for multimodal attachment tests
_TINY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _sample_questions() -> list[DecisionQuestion]:
    return MicroBenchDomain().decision_questions()


def _make_empty_board() -> InvestigationBoard:
    return InvestigationBoard(
        board_version=1,
        decision=BoardDecision(
            question_id="root_cause",
            question="Which component is the verified root cause?",
        ),
    )


class _FakeDecisionClient:
    """Deterministic fake `v1beta` client recording every `decisions.create` invocation."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.decisions = SimpleNamespace(create=self._create)

    def _create(
        self,
        *,
        model: str,
        input: Any,
        questions: dict[str, dict[str, Any]],
    ) -> Any:
        self.calls.append({"model": model, "input": input, "questions": questions})
        raw_text = input[0]["text"] if isinstance(input, list) else input
        parsed = json.loads(raw_text)

        # Enforce single-board invariant inside mock
        assert set(parsed.keys()) == {"board_version", "fields"}
        fields = parsed["fields"]

        required_clues = ("logs_retry_spike", "metrics_zscore_breach", "traces_error_cluster")
        all_payments_certified = all(
            fields.get(c, {}).get("status") == "certified"
            and fields.get(c, {}).get("value") == "payments-db"
            for c in required_clues
        )

        if all_payments_certified:
            rc_probs = {
                "payments-db": 0.94,
                "none_verified": 0.03,
                "checkout-svc": 0.01,
                "auth-svc": 0.01,
                "cache-redis": 0.01,
            }
            rc_choice = "payments-db"
            rc_conf = 0.94
            act_choice = "rollback"
            act_conf = 0.92
            act_probs = {"rollback": 0.92, "restart": 0.03, "scale_pool": 0.02, "escalate": 0.03}
            noul_val = 0.95
        else:
            rc_probs = {
                "none_verified": 0.91,
                "payments-db": 0.06,
                "checkout-svc": 0.01,
                "auth-svc": 0.01,
                "cache-redis": 0.01,
            }
            rc_choice = "none_verified"
            rc_conf = 0.91
            act_choice = "escalate"
            act_conf = 0.93
            act_probs = {"escalate": 0.93, "rollback": 0.04, "restart": 0.02, "scale_pool": 0.01}
            noul_val = 0.06

        answers: dict[str, Any] = {}
        for qid, qspec in questions.items():
            qtype = qspec["type"]
            if qtype == "choice":
                if qid == "root_cause":
                    full_probs = {opt: rc_probs.get(opt, 0.0) for opt in qspec["criteria"]}
                    answers[qid] = SimpleNamespace(
                        choice=rc_choice,
                        confidence=rc_conf,
                        probabilities=full_probs,
                    )
                else:
                    full_probs = {opt: act_probs.get(opt, 0.0) for opt in qspec["criteria"]}
                    answers[qid] = SimpleNamespace(
                        choice=act_choice,
                        confidence=act_conf,
                        probabilities=full_probs,
                    )
            elif qtype == "noul":
                answers[qid] = SimpleNamespace(
                    noul=noul_val,
                    confidence=max(noul_val, 1.0 - noul_val),
                )
            elif qtype == "score":
                labels = list(qspec["criteria"])
                answers[qid] = SimpleNamespace(
                    score=1.0,
                    confidence=0.88,
                    probabilities={lbl: (0.88 if idx == 0 else 0.12) for idx, lbl in enumerate(labels)},
                )

        usage = SimpleNamespace(
            total_input_tokens=120,
            total_output_tokens=0,
            total_thought_tokens=0,
            total_tokens=120,
        )
        return SimpleNamespace(answers=answers, usage=usage)


class _FakeInteractionsClient:
    """Deterministic fake `v1` Interactions client recording calls and kwargs."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.interactions = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        # Gemini 4 forbids legacy sampling / thinking_budget parameters
        gen_cfg = kwargs.get("generation_config") or {}
        for forbidden in ("temperature", "top_p", "top_k", "thinking_budget"):
            assert forbidden not in kwargs
            assert forbidden not in gen_cfg
        assert gen_cfg.get("thinking_level") in ("low", "medium", "high")
        assert kwargs.get("store") is False

        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("Unexpected extra interactions.create call.")
        return self._responses.pop(0)


# ----------------------------------------------------------------------
# 1. Pure Board Helpers & Explicit "unknown" Ablation Invariant
# ----------------------------------------------------------------------


def test_ablate_board_fields_keeps_keys_and_sets_explicit_unknown() -> None:
    board = InvestigationBoard(
        board_version=3,
        decision=BoardDecision(
            question_id="root_cause",
            question="Which component is the verified root cause?",
            answer="payments-db",
            confidence=0.95,
            certified_influence_share=1.0,
        ),
        fields={
            "logs_retry_spike": BoardField(
                value="payments-db",
                status="certified",
                sources=["note-1"],
                artifact_ids=["art-1"],
                influence=0.34,
            ),
            "metrics_zscore_breach": BoardField(
                value="payments-db",
                status="certified",
                sources=["note-2"],
            ),
        },
    )

    ablated = ablate_board_fields(board, ["logs_retry_spike"])

    # Original board is untouched; ablated board retains ALL keys
    assert set(ablated.fields.keys()) == {"logs_retry_spike", "metrics_zscore_breach"}
    assert board.fields["logs_retry_spike"].value == "payments-db"

    assert ablated.fields["logs_retry_spike"].value == "unknown"
    assert ablated.fields["logs_retry_spike"].status == "unknown"
    assert ablated.fields["logs_retry_spike"].sources == []
    assert ablated.fields["logs_retry_spike"].artifact_ids == []

    serialized = serialize_board_for_decision(ablated)
    parsed = json.loads(serialized)
    assert "decision" not in parsed
    assert "decisions" not in parsed
    assert parsed["fields"]["logs_retry_spike"] == {"value": "unknown", "status": "unknown"}
    assert parsed["fields"]["metrics_zscore_breach"] == {
        "value": "payments-db",
        "status": "certified",
    }


def test_serialize_board_shuffle_seed_permutes_field_order() -> None:
    board = _make_empty_board()
    for idx in range(6):
        board.fields[f"field_{idx}"] = BoardField(value=f"val_{idx}", status="certified")

    orders = {
        tuple(json.loads(serialize_board_for_decision(board, shuffle_seed=s))["fields"].keys())
        for s in range(5)
    }
    assert len(orders) > 1
    perturbed = perturb_board_field(board, "field_0", "override", status="uncertified")
    assert perturbed.fields["field_0"].value == "override"
    assert perturbed.fields["field_0"].status == "uncertified"


# ----------------------------------------------------------------------
# 2. DecisionProbeClient: One-Board-Per-Call, Influences, Shares & VoI
# ----------------------------------------------------------------------


def test_probe_board_one_board_per_call_and_pivotal_vs_distractor_influence() -> None:
    fake_client = _FakeDecisionClient()
    probe = DecisionProbeClient(client=fake_client)
    questions = _sample_questions()

    # 3 certified pivotal clues + 1 uncertified distractor + 1 unknown extra field
    board = InvestigationBoard(
        board_version=2,
        decision=BoardDecision(
            question_id="root_cause",
            question="Which component is the verified root cause?",
        ),
        fields={
            "logs_retry_spike": BoardField(
                value="payments-db", status="certified", sources=["n1"]
            ),
            "metrics_zscore_breach": BoardField(
                value="payments-db", status="certified", sources=["n2"]
            ),
            "traces_error_cluster": BoardField(
                value="payments-db", status="certified", sources=["n3"]
            ),
            "distractor_cache_noise": BoardField(
                value="cache-redis", status="uncertified", sources=["n4"]
            ),
            "auxiliary_host_note": BoardField(value="unknown", status="unknown"),
        },
    )

    candidate_vals = ["payments-db", "cache-redis"]
    result = probe.probe_board(
        board,
        questions,
        candidate_values=candidate_vals,
    )

    # Exact call count invariant: 1 base + 4 non-unknown ablations + (1 unknown * 2 candidates) = 7
    expected_ablations = 4
    expected_voi_variants = 1 * len(candidate_vals)
    assert len(fake_client.calls) == 1 + expected_ablations + expected_voi_variants
    assert result.api_calls == 1 + expected_ablations + expected_voi_variants
    assert result.telemetry.model_calls == 7
    assert result.telemetry.prompt_tokens == 7 * 120

    # Base evaluation & updated board decisions
    assert result.base_evaluation.answers["root_cause"].answer == "payments-db"
    assert result.updated_board.decision.answer == "payments-db"
    assert result.updated_board.decisions["recommended_action"].answer == "rollback"
    assert result.updated_board.decisions["sufficiently_verified"].answer == "yes"

    # Pivotal clues dominate influence (~1/3 each), distractor has 0.0 influence
    for clue in ("logs_retry_spike", "metrics_zscore_breach", "traces_error_cluster"):
        assert result.field_influences[clue] == pytest.approx(1.0 / 3.0, abs=1e-4)
        assert result.updated_board.fields[clue].influence == pytest.approx(1.0 / 3.0, abs=1e-4)

    assert result.field_influences["distractor_cache_noise"] == pytest.approx(0.0, abs=1e-6)
    assert result.certified_influence_share == pytest.approx(1.0, abs=1e-5)
    assert result.uncertified_influence_share == pytest.approx(0.0, abs=1e-5)
    assert result.unknown_voi_max == pytest.approx(0.0, abs=1e-5)


def test_probe_board_computes_high_unknown_voi_when_missing_third_clue() -> None:
    fake_client = _FakeDecisionClient()
    probe = DecisionProbeClient(client=fake_client)
    questions = _sample_questions()

    # 2 of 3 clues certified on payments-db; 3rd clue is "unknown"
    board = InvestigationBoard(
        board_version=1,
        decision=BoardDecision(
            question_id="root_cause",
            question="Which component is the verified root cause?",
        ),
        fields={
            "logs_retry_spike": BoardField(
                value="payments-db", status="certified", sources=["n1"]
            ),
            "metrics_zscore_breach": BoardField(
                value="payments-db", status="certified", sources=["n2"]
            ),
            "traces_error_cluster": BoardField(value="unknown", status="unknown"),
        },
    )

    cands = ["payments-db", "auth-svc"]
    result = probe.probe_board(board, questions, candidate_values=cands)

    # 1 base + 2 ablations + (1 unknown * 2 candidates) = 5 calls
    assert len(fake_client.calls) == 1 + 2 + 2
    assert result.api_calls == 5
    assert result.updated_board.decision.answer == "none_verified"
    assert result.updated_board.decisions["sufficiently_verified"].answer == "no"

    # Certifying `traces_error_cluster="payments-db"` flips root_cause -> high VoI!
    assert result.field_voi["traces_error_cluster"] > 0.80
    assert result.unknown_voi_max > 0.80
    assert result.updated_board.decision.unknown_voi_max > 0.80


def test_probe_board_contested_confidence_triggers_shuffled_repeat_and_bounded_score() -> None:
    calls: list[str] = []

    def _contested_create(*, model: str, input: Any, questions: dict[str, dict[str, Any]]) -> Any:
        assert model == "gemini-decision-1"
        calls.append(input)
        answers: dict[str, Any] = {}
        for qid, qspec in questions.items():
            if qspec["type"] == "choice":
                # Confidence 0.60 is inside (0.45, 0.85) contested band
                probs = {opt: 0.0 for opt in qspec["criteria"]}
                first_opt = next(iter(qspec["criteria"]))
                second_opt = list(qspec["criteria"])[1]
                probs[first_opt] = 0.60
                probs[second_opt] = 0.40
                answers[qid] = SimpleNamespace(
                    choice=first_opt, confidence=0.60, probabilities=probs
                )
            elif qspec["type"] == "score":
                labels = list(qspec["criteria"])
                answers[qid] = SimpleNamespace(
                    score=2.0,
                    confidence=0.75,
                    probabilities={labels[0]: 0.25, labels[1]: 0.75},
                )
        return SimpleNamespace(
            answers=answers,
            usage=SimpleNamespace(total_input_tokens=50, total_tokens=50),
        )

    fake_client = SimpleNamespace(decisions=SimpleNamespace(create=_contested_create))
    probe = DecisionProbeClient(client=fake_client)

    board = InvestigationBoard(
        board_version=1,
        decision=BoardDecision(question_id="root_cause", question="Root cause?"),
        fields={
            "f1": BoardField(value="a", status="unknown"),
            "f2": BoardField(value="b", status="unknown"),
        },
    )
    questions = [
        DecisionQuestion(
            question_id="root_cause",
            question="Root cause?",
            options=["payments-db", "none_verified"],
            question_type="multiple_choice",
        ),
        DecisionQuestion(
            question_id="severity_rubric",
            question="Severity level?",
            options=["low", "high"],
            question_type="bounded_score",
        ),
    ]

    res = probe.probe_board(board, questions, candidate_values=[])
    # 1 canonical + 1 shuffled repeat + 0 ablations + 0 VoI = 2 calls
    assert res.api_calls == 2
    assert len(calls) == 2
    assert res.base_evaluation.answers["severity_rubric"].answer == "high"
    assert res.base_evaluation.answers["severity_rubric"].raw_score == pytest.approx(2.0)


def test_propose_followup_tasks_sets_board_version_and_filters_verifiers() -> None:
    followup_json = json.dumps(
        {
            "queries": [
                {
                    "query_sql": "SELECT service, COUNT(*) FROM traces WHERE status='ERROR' GROUP BY 1;",
                    "hypothesis_family": "downstream_trace_timeout_cluster",
                    "expected_signal": "Check missing trace error cluster",
                    "target_fields": ["traces_error_cluster"],
                    "verifier_name": "verify_trace_error_cluster",
                },
                {
                    "query_sql": "SELECT service, COUNT(*) FROM metrics GROUP BY 1",
                    "hypothesis_family": "metric_check",
                    "expected_signal": "Exploratory check",
                    "target_fields": ["metrics_zscore_breach"],
                    "verifier_name": "nonexistent_verifier",
                },
            ]
        }
    )
    fake_resp = SimpleNamespace(
        output_text=followup_json,
        usage=SimpleNamespace(total_input_tokens=150, total_output_tokens=60, total_tokens=210),
    )
    fake_client = _FakeInteractionsClient([fake_resp])
    provider = GeminiModelProvider(client=fake_client)

    board = _make_empty_board()
    board.board_version = 4
    specs, tel = provider.propose_followup_tasks(
        board=board,
        high_voi_fields=["traces_error_cluster"],
        schema_description="service_logs, metrics, traces",
        available_verifiers=["verify_trace_error_cluster"],
    )

    assert len(specs) == 2
    assert specs[0].verifier_name == "verify_trace_error_cluster"
    assert specs[0].requested_under == 4
    assert specs[1].verifier_name is None
    assert tel.model_calls == 1


# ----------------------------------------------------------------------
# 3. GeminiModelProvider: ARES `solve(...)` Processor & Multimodal Patch
# ----------------------------------------------------------------------


def test_make_propose_processor_executes_inside_ares_solve() -> None:
    proposal_json = json.dumps(
        {
            "queries": [
                {
                    "query_sql": "SELECT downstream_service, COUNT(*) AS c FROM service_logs WHERE level = 'ERROR' GROUP BY 1;",
                    "hypothesis_family": "upstream_retry_spike",
                    "expected_signal": ">=12 upstream retry errors",
                    "target_fields": ["logs_retry_spike"],
                    "verifier_name": "verify_upstream_retry_spike",
                }
            ]
        }
    )
    fake_resp = SimpleNamespace(
        output_text=proposal_json,
        usage=SimpleNamespace(
            total_input_tokens=210,
            total_output_tokens=85,
            total_thought_tokens=40,
            total_tokens=335,
        ),
    )
    fake_client = _FakeInteractionsClient([fake_resp])
    provider = GeminiModelProvider(client=fake_client)

    domain = MicroBenchDomain()
    proc = provider.make_propose_processor(
        schema_description=domain.schema_description(),
        available_verifiers=["verify_upstream_retry_spike", "verify_metric_zscore_breach"],
    )

    task = TaskSpec(
        task_id="propose_retries",
        task_type="propose",
        requested_under=2,
        context={
            "hypothesis_family": "upstream_retry_spike",
            "max_queries": 2,
            "thinking_level": "minimal",  # Should normalize cleanly to "low"
        },
    )
    report = solve(tasks=[task], processors={"propose": proc}, max_concurrent=2)

    assert report.errors == {}
    assert report.usage.completed_tasks == 1
    assert report.usage.model_calls == 1
    assert report.usage.total_prompt_tokens == 210
    assert report.usage.total_completion_tokens == 85

    emitted_specs = report.graph.nodes["propose_retries"].output
    assert isinstance(emitted_specs, list) and len(emitted_specs) == 1
    spec0 = emitted_specs[0]
    assert isinstance(spec0, HypothesisQuerySpec)
    assert not spec0.query_sql.endswith(";")
    assert spec0.verifier_name == "verify_upstream_retry_spike"
    assert spec0.requested_under == 2


def test_propose_board_patch_attaches_multimodal_png_and_filters_sources(
    tmp_path: Path,
) -> None:
    png_path = tmp_path / "candidate_slice.png"
    png_path.write_bytes(_TINY_PNG_BYTES)

    art = ArtifactRef(
        artifact_id="art-plot-01",
        kind="plot_png",
        path=str(png_path),
        sha256="deadbeef",
        mime_type="image/png",
        produced_by_verifier=True,
    )
    note = Note(
        note_id="note-3way-01",
        certification="CERTIFIED",
        verifier_name="verify_three_way_root_cause",
        summary="CERTIFIED 3-way root cause(s): ['payments-db']",
        evidence={"verified_root_causes": ["payments-db"]},
        artifacts=[art],
        row_count=1,
    )

    patch_json = json.dumps(
        {
            "updates": [
                {
                    "field_name": "three_way_root_cause",
                    "value_json": json.dumps("payments-db"),
                    "sources": ["note-3way-01", "hallucinated-note-id"],
                    "artifact_ids": ["art-plot-01", "bogus-artifact"],
                },
                {
                    "field_name": "ungrounded_field",
                    "value_json": "invented",
                    "sources": ["hallucinated-only"],
                    "artifact_ids": [],
                },
            ],
            "rationale": "3-way verifier certified payments-db (note-3way-01).",
        }
    )
    fake_resp = SimpleNamespace(
        output_text=patch_json,
        usage=SimpleNamespace(
            total_input_tokens=400,
            total_output_tokens=95,
            total_thought_tokens=60,
            total_tokens=555,
        ),
    )
    fake_client = _FakeInteractionsClient([fake_resp])
    provider = GeminiModelProvider(client=fake_client)

    patch, telemetry = provider.propose_board_patch(
        board=_make_empty_board(),
        new_notes=[note],
        schema_description="service_logs, metrics, traces",
    )

    assert len(fake_client.calls) == 1
    sent_input = fake_client.calls[0]["input"]
    assert isinstance(sent_input, list) and len(sent_input) == 2
    assert sent_input[0]["type"] == "text"
    assert sent_input[1] == {
        "type": "image",
        "data": base64.b64encode(_TINY_PNG_BYTES).decode("ascii"),
        "mime_type": "image/png",
    }

    assert patch == {
        "updates": {
            "three_way_root_cause": {
                "value": "payments-db",
                "sources": ["note-3way-01"],
                "artifact_ids": ["art-plot-01"],
            }
        },
        "rationale": "3-way verifier certified payments-db (note-3way-01).",
    }
    assert telemetry.model_calls == 1
    assert telemetry.thought_tokens == 60


def test_run_carbon_sql_baseline_stateless_tool_loop() -> None:
    fc_step = SimpleNamespace(
        type="function_call",
        id="call-1",
        name="execute_readonly_sql",
        arguments={"sql": "SELECT service, COUNT(*) AS c FROM traces WHERE status='ERROR' GROUP BY 1;"},
        model_dump=lambda: {
            "type": "function_call",
            "id": "call-1",
            "name": "execute_readonly_sql",
            "arguments": {
                "sql": "SELECT service, COUNT(*) AS c FROM traces WHERE status='ERROR' GROUP BY 1;"
            },
        },
    )
    turn1 = SimpleNamespace(
        steps=[fc_step],
        output_text="",
        usage=SimpleNamespace(total_input_tokens=100, total_output_tokens=30, total_tokens=130),
    )
    turn2 = SimpleNamespace(
        steps=[],
        output_text=json.dumps(
            {
                "root_cause": "payments-db",
                "recommended_action": "rollback",
                "sufficiently_verified": "yes",
                "confidence": 0.89,
                "rationale": "Verified across logs, metrics, and traces.",
            }
        ),
        usage=SimpleNamespace(total_input_tokens=180, total_output_tokens=45, total_tokens=225),
    )

    fake_client = _FakeInteractionsClient([turn1, turn2])
    provider = GeminiModelProvider(client=fake_client)
    executed_sql: list[str] = []

    def _exec(sql: str) -> dict[str, Any]:
        executed_sql.append(sql)
        return {"columns": ["service", "c"], "rows": [{"service": "payments-db", "c": 16}], "row_count": 1}

    res = provider.run_carbon_sql_baseline(
        schema_description="service_logs, metrics, traces",
        decision_questions=_sample_questions(),
        sql_executor=_exec,
        max_queries=4,
    )

    assert len(executed_sql) == 1
    assert res["root_cause"] == "payments-db"
    assert res["recommended_action"] == "rollback"
    assert res["sufficiently_verified"] == "yes"
    assert res["confidence"] == pytest.approx(0.89)
    assert res["telemetry"].model_calls == 2
    assert res["telemetry"].prompt_tokens == 280


# ----------------------------------------------------------------------
# 4. Live API Smoke Test (runs automatically when GEMINI_API_KEY is set)
# ----------------------------------------------------------------------


load_dotenv()
_HAS_LIVE_API_KEY = bool(os.environ.get("GEMINI_API_KEY"))


@pytest.mark.skipif(not _HAS_LIVE_API_KEY, reason="GEMINI_API_KEY not configured in environment.")
def test_live_gemini_hydrogen_carbon_and_decision_probe_smoke(tmp_path: Path) -> None:
    domain = MicroBenchDomain()
    schema = domain.schema_description()
    questions = domain.decision_questions()
    verifiers = [
        "verify_upstream_retry_spike",
        "verify_metric_zscore_breach",
        "verify_trace_error_cluster",
        "verify_three_way_root_cause",
    ]

    provider = GeminiModelProvider()

    # 1. Live gemini-4-hydrogen hypothesis proposal
    specs, h_tel = provider.propose_hypotheses(
        schema_description=schema,
        hypothesis_family="upstream_retry_spike",
        available_verifiers=verifiers,
        max_queries=2,
        thinking_level="low",
    )
    assert len(specs) >= 1
    assert "select" in specs[0].query_sql.lower()
    assert "<data_dir>" not in specs[0].query_sql
    assert h_tel.model_calls == 1
    assert h_tel.prompt_tokens > 0

    # 2. Live gemini-4-carbon multimodal board patch synthesis
    plot_path = tmp_path / "payments_db_slice.png"
    plot_path.write_bytes(_TINY_PNG_BYTES)
    art = ArtifactRef(
        artifact_id="art-live-01",
        kind="plot_png",
        path=str(plot_path),
        sha256="abcd1234",
        mime_type="image/png",
        produced_by_verifier=True,
    )
    notes = [
        Note(
            note_id="note-logs-1",
            certification="CERTIFIED",
            verifier_name="verify_upstream_retry_spike",
            summary="Found 1 (service, shard) retry spike(s).",
            evidence={
                "candidates": [
                    {
                        "service": "payments-db",
                        "shard_id": "shard-02",
                        "retry_log_count": 18,
                        "distinct_traces": 18,
                    }
                ]
            },
            row_count=1,
        ),
        Note(
            note_id="note-metrics-1",
            certification="CERTIFIED",
            verifier_name="verify_metric_zscore_breach",
            summary="Found 1 metric z-score breach(es).",
            evidence={
                "candidates": [
                    {
                        "service": "payments-db",
                        "shard_id": "shard-02",
                        "metric_name": "lock_wait_ms",
                        "breach_buckets": 6,
                        "max_zscore": 6.8,
                    }
                ]
            },
            row_count=1,
        ),
        Note(
            note_id="note-traces-1",
            certification="CERTIFIED",
            verifier_name="verify_trace_error_cluster",
            summary="Found 1 downstream trace error cluster(s).",
            evidence={
                "candidates": [
                    {
                        "service": "payments-db",
                        "shard_id": "shard-02",
                        "error_spans": 18,
                        "distinct_traces": 18,
                        "max_duration_ms": 1420.0,
                    }
                ]
            },
            artifacts=[art],
            row_count=1,
        ),
    ]

    patch, c_tel = provider.propose_board_patch(
        board=_make_empty_board(),
        new_notes=notes,
        schema_description=schema,
        include_images=True,
        thinking_level="low",
    )
    assert c_tel.model_calls == 1
    assert len(patch["updates"]) >= 3
    for upd in patch["updates"].values():
        assert set(upd["sources"]).issubset({"note-logs-1", "note-metrics-1", "note-traces-1"})

    # 3. Live gemini-decision-1 probe on 3-clue CERTIFIED board vs ablated 2-of-3 board
    full_board = InvestigationBoard(
        board_version=2,
        decision=BoardDecision(
            question_id="root_cause",
            question="Which component is the verified root cause?",
        ),
        fields={
            "logs_retry_spike": BoardField(
                value="payments-db", status="certified", sources=["note-logs-1"]
            ),
            "metrics_zscore_breach": BoardField(
                value="payments-db", status="certified", sources=["note-metrics-1"]
            ),
            "traces_error_cluster": BoardField(
                value="payments-db", status="certified", sources=["note-traces-1"]
            ),
            "candidate_remediation": BoardField(
                value="rollback", status="certified", sources=["note-metrics-1"]
            ),
        },
    )

    probe_client = DecisionProbeClient()
    full_probe = probe_client.probe_board(
        full_board,
        questions,
        candidate_values=[],
    )
    assert full_probe.updated_board.decision.answer == "payments-db"
    assert full_probe.updated_board.decisions["sufficiently_verified"].answer == "yes"
    assert full_probe.certified_influence_share > 0.80
    assert full_probe.uncertified_influence_share == pytest.approx(0.0, abs=1e-5)

    # Ablate traces_error_cluster -> root_cause must drop to none_verified with high unknown VoI
    incomplete_board = ablate_board_fields(full_board, ["traces_error_cluster"])
    inc_probe = probe_client.probe_board(
        incomplete_board,
        questions,
        candidate_values=["payments-db", "cache-redis"],
    )
    assert inc_probe.updated_board.decision.answer == "none_verified"
    assert inc_probe.updated_board.decisions["sufficiently_verified"].answer == "no"
    assert inc_probe.unknown_voi_max > 0.30
