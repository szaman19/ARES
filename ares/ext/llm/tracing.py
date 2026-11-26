"""Opt-in artifact-backed LLM tracing wrapper for ARES tasks."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from ares.ext.llm.base import LLMClient, LLMMessage, LLMResponse


@runtime_checkable
class _ArtifactRefLike(Protocol):
    ref_id: str


@runtime_checkable
class _ArtifactStoreLike(Protocol):
    def put_json(self, obj: Any, *, name: str | None = None) -> _ArtifactRefLike: ...


@runtime_checkable
class _TaskContextLike(Protocol):
    store: _ArtifactStoreLike
    task_id: str | None
    run_id: str | None

    def record_usage(
        self,
        *,
        prompt_tokens: int = ...,
        completion_tokens: int = ...,
        total_tokens: int | None = ...,
        llm_calls: int = ...,
        cost_usd: float = ...,
    ) -> Any: ...


class TracedLLMClient:
    """``LLMClient`` wrapper that records prompts, responses, reasoning, and usage to ``ctx``."""

    def __init__(
        self,
        inner: LLMClient,
        ctx: _TaskContextLike,
        *,
        model_label: str | None = None,
        cost_per_1k_prompt_tokens: float = 0.0,
        cost_per_1k_completion_tokens: float = 0.0,
        emit_events: bool = True,
    ) -> None:
        self.inner = inner
        self.ctx = ctx
        self.model_label = model_label
        self.cost_per_1k_prompt_tokens = float(cost_per_1k_prompt_tokens)
        self.cost_per_1k_completion_tokens = float(cost_per_1k_completion_tokens)
        self.emit_events = emit_events
        self._lock = threading.Lock()
        self._call_counter = 0
        self._trace_ref_ids: list[str] = []

    @property
    def trace_ref_ids(self) -> list[str]:
        """Return a thread-safe snapshot of artifact ``ref_id``s written by this client."""
        with self._lock:
            return list(self._trace_ref_ids)

    def _resolve_model(self) -> str:
        if self.model_label is not None:
            return self.model_label
        inner_model = getattr(self.inner, "model", None)
        if inner_model:
            return str(inner_model)
        return type(self.inner).__name__

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost = (prompt_tokens / 1000.0) * self.cost_per_1k_prompt_tokens + (
            completion_tokens / 1000.0
        ) * self.cost_per_1k_completion_tokens
        return round(cost, 10)

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Execute ``inner.complete`` and persist an auditable trace record to ``ctx.store``."""
        with self._lock:
            self._call_counter += 1
            call_index = self._call_counter

        resolved_model = self._resolve_model()
        serialized_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        t0 = time.perf_counter()
        response: LLMResponse | None = None
        error_str: str | None = None
        status = "ok"

        try:
            response = self.inner.complete(
                messages,
                system=system,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response
        except Exception as exc:
            status = "error"
            error_str = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            if response is not None:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens or (
                    prompt_tokens + completion_tokens
                )
                latency_ms = (
                    response.usage.latency_ms if response.usage.latency_ms > 0.0 else wall_ms
                )
                cost_usd = self._compute_cost(prompt_tokens, completion_tokens)
                parsed_payload: Any = response.parsed
                if isinstance(parsed_payload, BaseModel):
                    parsed_payload = parsed_payload.model_dump(mode="json")
                response_text: str | None = response.text
                reasoning: str | None = response.reasoning
            else:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                latency_ms = wall_ms
                cost_usd = 0.0
                parsed_payload = None
                response_text = None
                reasoning = None

            record: dict[str, Any] = {
                "call_index": call_index,
                "task_id": getattr(self.ctx, "task_id", None),
                "run_id": getattr(self.ctx, "run_id", None),
                "model": resolved_model,
                "system": system,
                "messages": serialized_messages,
                "response_model": (
                    response_model.__name__ if response_model is not None else None
                ),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "status": status,
                "response_text": response_text,
                "reasoning": reasoning,
                "parsed": parsed_payload,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "latency_ms": latency_ms,
                },
                "cost_usd": cost_usd,
                "error": error_str,
            }

            task_label = getattr(self.ctx, "task_id", None) or "unknown"
            trace_name = f"llm_trace_{task_label}_{call_index}.json"
            ref = self.ctx.store.put_json(record, name=trace_name)
            with self._lock:
                self._trace_ref_ids.append(ref.ref_id)

            if response is not None:
                self.ctx.record_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    llm_calls=1,
                    cost_usd=cost_usd,
                )
            else:
                self.ctx.record_usage(llm_calls=1)

            if self.emit_events and hasattr(self.ctx, "emit"):
                try:
                    self.ctx.emit(
                        "trace",
                        {
                            "kind": "llm_call",
                            "trace_ref_id": ref.ref_id,
                            "status": status,
                            "model": resolved_model,
                            "latency_ms": latency_ms,
                        },
                    )
                except Exception:
                    pass
