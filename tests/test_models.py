"""Offline unit tests (plus opt-in live smoke test) for `ares/models/gemini.py`."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ares.contracts import (
    ArtifactRef,
    BoardConclusion,
    HypothesisSpec,
    InvestigationBoard,
    Note,
    SearchQuestion,
)
from ares.engine import TaskSpec, solve
from ares.models import GeminiModelProvider
from examples.incident_rca.domain import MicroBenchDomain

_TINY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _sample_questions() -> list[SearchQuestion]:
    return MicroBenchDomain().search_questions()


def _make_empty_board() -> InvestigationBoard:
    return InvestigationBoard(
        board_version=1,
        conclusion=BoardConclusion(
            question_id="root_cause",
            question="Which component is the verified root cause?",
        ),
    )


class _FakeInteractionsClient:
    """Deterministic fake `v1` Interactions client recording calls and kwargs."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.interactions = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
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


def test_propose_followup_tasks_sets_board_version_and_filters_verifiers() -> None:
    followup_json = json.dumps(
        {
            "hypotheses": [
                {
                    "query": "SELECT service, COUNT(*) FROM traces WHERE status='ERROR' GROUP BY 1;",
                    "params_json": "{}",
                    "hypothesis_family": "downstream_trace_timeout_cluster",
                    "expected_signal": "Check missing trace error cluster",
                    "target_fields": ["traces_error_cluster"],
                    "verifier_name": "verify_trace_error_cluster",
                },
                {
                    "query": "SELECT service, COUNT(*) FROM metrics GROUP BY 1",
                    "params_json": "{}",
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
        target_gap_fields=["traces_error_cluster"],
        environment_description="service_logs, metrics, traces",
        available_verifiers=["verify_trace_error_cluster"],
    )

    assert len(specs) == 2
    assert specs[0].verifier_name == "verify_trace_error_cluster"
    assert specs[0].requested_under == 4
    assert specs[1].verifier_name is None
    assert tel.model_calls == 1


def test_make_propose_processor_executes_inside_ares_solve() -> None:
    proposal_json = json.dumps(
        {
            "hypotheses": [
                {
                    "query": "SELECT downstream_service, COUNT(*) AS c FROM service_logs WHERE level = 'ERROR' GROUP BY 1;",
                    "params_json": "{}",
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
        environment_description=domain.environment_description(),
        magnet_rules=domain.magnet_rules(),
        pivotal_fields=domain.pivotal_fields(),
        available_verifiers=["verify_upstream_retry_spike", "verify_metric_zscore_breach"],
    )

    task = TaskSpec(
        task_id="propose_retries",
        task_type="propose",
        requested_under=2,
        context={
            "hypothesis_family": "upstream_retry_spike",
            "max_queries": 2,
            "thinking_level": "minimal",
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
    assert isinstance(spec0, HypothesisSpec)
    assert not spec0.query.endswith(";")
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
        environment_description="service_logs, metrics, traces",
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


def test_run_single_agent_baseline_stateless_tool_loop() -> None:
    fc_step = SimpleNamespace(
        type="function_call",
        id="call-1",
        name="execute_readonly_query",
        arguments={"query": "SELECT service, COUNT(*) AS c FROM traces WHERE status='ERROR' GROUP BY 1;"},
        model_dump=lambda: {
            "type": "function_call",
            "id": "call-1",
            "name": "execute_readonly_query",
            "arguments": {
                "query": "SELECT service, COUNT(*) AS c FROM traces WHERE status='ERROR' GROUP BY 1;"
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
                "answers": [
                    {"question_id": "root_cause", "answer": "payments-db"},
                    {"question_id": "recommended_action", "answer": "rollback"},
                    {"question_id": "sufficiently_verified", "answer": "yes"},
                ],
                "confidence": 0.89,
                "rationale": "Verified across logs, metrics, and traces.",
            }
        ),
        usage=SimpleNamespace(total_input_tokens=180, total_output_tokens=45, total_tokens=225),
    )

    fake_client = _FakeInteractionsClient([turn1, turn2])
    provider = GeminiModelProvider(client=fake_client)
    executed_queries: list[str] = []

    def _exec(q: str) -> dict[str, Any]:
        executed_queries.append(q)
        return {"columns": ["service", "c"], "rows": [{"service": "payments-db", "c": 16}], "row_count": 1}

    res = provider.run_single_agent_baseline(
        environment_description="service_logs, metrics, traces",
        search_questions=_sample_questions(),
        query_executor=_exec,
        max_queries=4,
    )

    assert len(executed_queries) == 1
    assert res["answers"]["root_cause"] == "payments-db"
    assert res["answers"]["recommended_action"] == "rollback"
    assert res["answers"]["sufficiently_verified"] == "yes"
    assert res["confidence"] == pytest.approx(0.89)
    assert res["telemetry"].model_calls == 2
    assert res["telemetry"].prompt_tokens == 280


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("ARES_RUN_LIVE_TESTS") == "1"),
    reason="Live Gemini API tests are opt-in (set GEMINI_API_KEY and ARES_RUN_LIVE_TESTS=1)",
)
def test_live_gemini_hydrogen_hypothesis_smoke() -> None:
    domain = MicroBenchDomain()
    provider = GeminiModelProvider()
    specs, h_tel = provider.propose_hypotheses(
        environment_description=domain.environment_description(),
        magnet_rules=domain.magnet_rules(),
        pivotal_fields=domain.pivotal_fields(),
        hypothesis_family="upstream_retry_spike",
        available_verifiers=[
            "verify_upstream_retry_spike",
            "verify_metric_zscore_breach",
            "verify_trace_error_cluster",
            "verify_three_way_root_cause",
        ],
        max_queries=2,
        thinking_level="low",
    )
    assert len(specs) >= 1
    assert h_tel.model_calls == 1
    assert h_tel.prompt_tokens > 0
