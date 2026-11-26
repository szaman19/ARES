"""Integration tests for ``TracedLLMClient`` with ``StateMachine``, ``ArtifactStore``, and ``CheckpointStore``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from ares.ext.llm import (
    ContextLengthExceededError,
    LLMClient,
    LLMMessage,
    LLMResponse,
    LLMUsage,
    OpenAICompatClient,
    TracedLLMClient,
)
from ares.machine import StateMachine, TaskContext
from ares.store import ArtifactStore, CheckpointStore
from ares.types import Done, Fail, Task, TaskSpec


class _Verdict(BaseModel):
    label: str
    score: float


def test_traced_llm_client_persists_artifacts_and_accumulates_usage(tmp_path: Path) -> None:
    """Verify traces, reasoning, parsed outputs, and usage/cost accumulate across a real run."""
    machine = StateMachine()
    captured_refs: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if "response_format" in body:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": (
                                    "<think>evaluating verdict</think>\n"
                                    '{"label": "valid", "score": 0.9}'
                                ),
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 200,
                        "completion_tokens": 100,
                        "total_tokens": 300,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "step-1-complete",
                            "reasoning_content": "initial chain of thought",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        )

    @machine.state("solve")
    def solve(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done:
        raw_client = OpenAICompatClient(
            model="test-reasoner",
            transport=httpx.MockTransport(handler),
        )
        llm = TracedLLMClient(
            raw_client,
            ctx,
            cost_per_1k_prompt_tokens=0.002,
            cost_per_1k_completion_tokens=0.004,
        )
        assert isinstance(llm, LLMClient)

        with raw_client:
            r1 = llm.complete(
                [LLMMessage(role="user", content="first step")],
                system="Be concise.",
                temperature=0.1,
                max_tokens=128,
            )
            r2 = llm.complete(
                [LLMMessage(role="user", content="final verdict")],
                response_model=_Verdict,
            )

        captured_refs.extend(llm.trace_ref_ids)
        return Done(output={"first": r1.text, "verdict": r2.parsed.model_dump()})

    result = machine.run(TaskSpec(state="solve"), run_dir=tmp_path)
    assert result.status == "done"
    assert result.output == {"first": "step-1-complete", "verdict": {"label": "valid", "score": 0.9}}

    # Cumulative RunResult usage: (100+200) prompt, (50+100) completion, 450 total, 2 calls, $0.0012
    assert result.usage.prompt_tokens == 300
    assert result.usage.completion_tokens == 150
    assert result.usage.total_tokens == 450
    assert result.usage.llm_calls == 2
    assert result.usage.cost_usd == pytest.approx(0.0012)

    assert len(captured_refs) == 2

    # Reload artifacts & checkpoints from disk to confirm durability
    reloaded_store = ArtifactStore(tmp_path / "artifacts")
    trace_1 = reloaded_store.get_json(captured_refs[0])
    trace_2 = reloaded_store.get_json(captured_refs[1])

    assert trace_1["call_index"] == 1
    assert trace_1["task_id"] == result.root_task_id
    assert trace_1["run_id"] == result.run_id
    assert trace_1["model"] == "test-reasoner"
    assert trace_1["system"] == "Be concise."
    assert trace_1["messages"] == [{"role": "user", "content": "first step"}]
    assert trace_1["response_model"] is None
    assert trace_1["temperature"] == 0.1
    assert trace_1["max_tokens"] == 128
    assert trace_1["status"] == "ok"
    assert trace_1["response_text"] == "step-1-complete"
    assert trace_1["reasoning"] == "initial chain of thought"
    assert trace_1["parsed"] is None
    assert trace_1["usage"]["prompt_tokens"] == 100
    assert trace_1["usage"]["completion_tokens"] == 50
    assert trace_1["usage"]["total_tokens"] == 150
    assert trace_1["usage"]["latency_ms"] >= 0.0
    assert trace_1["cost_usd"] == pytest.approx(0.0004)
    assert trace_1["error"] is None

    assert trace_2["call_index"] == 2
    assert trace_2["response_model"] == "_Verdict"
    assert trace_2["status"] == "ok"
    assert trace_2["reasoning"] == "evaluating verdict"
    assert trace_2["parsed"] == {"label": "valid", "score": 0.9}
    assert trace_2["cost_usd"] == pytest.approx(0.0008)
    assert trace_2["error"] is None

    checkpoint = CheckpointStore(tmp_path)
    assert checkpoint.load_state() is not None
    assert len(checkpoint.load_events()) > 0


def test_traced_llm_client_persists_error_trace_and_usage_on_exception(tmp_path: Path) -> None:
    """Persist failed LLM call trace and increment ``llm_calls`` even when inner raises."""
    machine = StateMachine()
    captured_refs: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "This model's maximum context length is 4096 tokens."}},
        )

    @machine.state("overflow")
    def overflow(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Fail:
        raw_client = OpenAICompatClient(
            model="small-ctx-model",
            transport=httpx.MockTransport(handler),
        )
        llm = TracedLLMClient(raw_client, ctx, model_label="custom-label")
        try:
            llm.complete([LLMMessage(role="user", content="too many tokens")])
        except ContextLengthExceededError as exc:
            captured_refs.extend(llm.trace_ref_ids)
            return Fail(error=str(exc))
        return Fail(error="expected overflow")

    result = machine.run(TaskSpec(state="overflow"), run_dir=tmp_path)
    assert result.status == "failed"
    assert result.usage.llm_calls == 1
    assert result.usage.prompt_tokens == 0
    assert result.usage.total_tokens == 0
    assert len(captured_refs) == 1

    reloaded_store = ArtifactStore(tmp_path / "artifacts")
    error_trace = reloaded_store.get_json(captured_refs[0])
    assert error_trace["status"] == "error"
    assert error_trace["model"] == "custom-label"
    assert error_trace["response_text"] is None
    assert error_trace["reasoning"] is None
    assert error_trace["parsed"] is None
    assert error_trace["cost_usd"] == 0.0
    assert error_trace["usage"]["latency_ms"] >= 0.0
    assert "ContextLengthExceededError" in error_trace["error"]


def test_traced_llm_client_emits_trace_event_when_supported() -> None:
    """Emit ``trace`` events when ``ctx.emit`` accepts custom event types and ``emit_events=True``."""
    emitted: list[tuple[str, dict[str, Any]]] = []
    store = ArtifactStore(None)

    class _CustomCtx:
        def __init__(self) -> None:
            self.store = store
            self.task_id = "task_9"
            self.run_id = "run_9"
            self.usage_calls: list[dict[str, Any]] = []

        def record_usage(self, **kwargs: Any) -> None:
            self.usage_calls.append(kwargs)

        def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
            emitted.append((event_type, payload or {}))

    class _StubInner:
        model = "stub-model"

        def complete(
            self,
            messages: Any,
            *,
            system: str | None = None,
            response_model: Any = None,
            temperature: float = 0.0,
            max_tokens: int | None = None,
        ) -> LLMResponse:
            return LLMResponse(
                text="hello",
                reasoning="thought",
                usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, latency_ms=4.2),
            )

    ctx = _CustomCtx()
    traced = TracedLLMClient(_StubInner(), ctx, emit_events=True)
    traced.complete([LLMMessage(role="user", content="hi")])

    assert len(emitted) == 1
    ev_type, ev_payload = emitted[0]
    assert ev_type == "trace"
    assert ev_payload == {
        "kind": "llm_call",
        "trace_ref_id": traced.trace_ref_ids[0],
        "status": "ok",
        "model": "stub-model",
        "latency_ms": 4.2,
    }


def test_llm_boundary_invariant_passes_with_tracing() -> None:
    """Confirm ``ares/ext/llm/tracing.py`` maintains zero core/domain coupling."""
    from tests.test_llm_openai_compat import test_ares_llm_boundary_no_core_or_domain_coupling

    test_ares_llm_boundary_no_core_or_domain_coupling()
