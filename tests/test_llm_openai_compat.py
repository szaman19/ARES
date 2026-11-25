"""Offline tests for ``ares.llm`` and ``OpenAICompatClient`` using ``httpx.MockTransport``."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from ares.llm import (
    ContextLengthExceededError,
    LLMClient,
    LLMMessage,
    LLMResponse,
    LLMUsage,
    OpenAICompatClient,
)


class _StructuredAnswer(BaseModel):
    answer: str
    confidence: float


def test_plain_text_completion_usage_auth_and_url_normalization() -> None:
    """Verify plain text completion, usage extraction, auth header, and URL normalization."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            },
        )

    for base_url in ("http://localhost:11434/v1", "http://localhost:11434/v1/"):
        captured.clear()
        client = OpenAICompatClient(
            base_url=base_url,
            model="test-model",
            api_key="secret-key",
            reasoning_effort="medium",
            transport=httpx.MockTransport(handler),
        )
        assert isinstance(client, LLMClient)

        resp = client.complete(
            [LLMMessage(role="user", content="ping")],
            system="You are terse.",
            temperature=0.2,
            max_tokens=64,
        )

        assert isinstance(resp, LLMResponse)
        assert resp.text == "pong"
        assert resp.parsed is None
        assert isinstance(resp.usage, LLMUsage)
        assert resp.usage.prompt_tokens == 11
        assert resp.usage.completion_tokens == 3
        assert resp.usage.total_tokens == 14
        assert resp.usage.latency_ms >= 0.0
        assert resp.raw["id"] == "chatcmpl-1"

        assert len(captured) == 1
        req = captured[0]
        assert str(req.url) == "http://localhost:11434/v1/chat/completions"
        assert req.headers["authorization"] == "Bearer secret-key"
        body = json.loads(req.content.decode("utf-8"))
        assert body["model"] == "test-model"
        assert body["temperature"] == 0.2
        assert body["max_tokens"] == 64
        assert body["reasoning_effort"] == "medium"
        assert body["messages"] == [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "ping"},
        ]
        assert "response_format" not in body


def test_structured_response_model_json_schema() -> None:
    """Verify ``response_model`` sends OpenAI ``json_schema`` and validates output."""
    captured_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"answer": "42", "confidence": 0.95}',
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    client = OpenAICompatClient(
        model="structured-model",
        transport=httpx.MockTransport(handler),
    )
    resp = client.complete(
        [LLMMessage(role="user", content="What is the answer?")],
        response_model=_StructuredAnswer,
    )

    assert len(captured_bodies) == 1
    rf = captured_bodies[0]["response_format"]
    assert rf == {
        "type": "json_schema",
        "json_schema": {
            "name": "_StructuredAnswer",
            "schema": _StructuredAnswer.model_json_schema(),
            "strict": True,
        },
    }
    assert isinstance(resp.parsed, _StructuredAnswer)
    assert resp.parsed.answer == "42"
    assert resp.parsed.confidence == 0.95
    assert resp.usage.total_tokens == 30


@pytest.mark.parametrize(
    ("status_code", "error_body"),
    [
        (400, {"error": {"message": "response_format json_schema is not supported"}}),
        (422, {"detail": "Unsupported parameter: response_format"}),
    ],
)
def test_fallback_when_response_format_rejected_with_fence_stripping(
    status_code: int,
    error_body: dict[str, Any],
) -> None:
    """Retry once without ``response_format`` and strip markdown JSON fences."""
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(status_code, json=error_body)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```json\n{"answer": "ok", "confidence": 0.8}\n```',
                        },
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
            },
        )

    client = OpenAICompatClient(transport=httpx.MockTransport(handler))
    resp = client.complete(
        [LLMMessage(role="user", content="Give structured answer")],
        response_model=_StructuredAnswer,
    )

    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]
    assert "_StructuredAnswer" in calls[1]["messages"][-1]["content"]
    assert '"answer"' in calls[1]["messages"][-1]["content"]
    assert isinstance(resp.parsed, _StructuredAnswer)
    assert resp.parsed == _StructuredAnswer(answer="ok", confidence=0.8)


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (400, "This model's maximum context length is 8192 tokens."),
        (413, "Payload too large: prompt is too long for context window."),
        (400, "Request has too many tokens; exceeds the available context."),
        (422, "Input exceeds token limit of 4096."),
    ],
)
def test_context_length_exceeded_raises_error(status_code: int, message: str) -> None:
    """Raise ``ContextLengthExceededError`` on 400/413/422 context-limit responses."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code, json={"error": {"message": message}})

    client = OpenAICompatClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ContextLengthExceededError):
        client.complete(
            [LLMMessage(role="user", content="x" * 1000)],
            response_model=_StructuredAnswer,
        )
    assert call_count == 1


def test_http_500_raises_status_error() -> None:
    """Non-context server errors still surface as ``httpx.HTTPStatusError``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal server error"})

    client = OpenAICompatClient(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        client.complete([LLMMessage(role="user", content="hello")])


def test_ares_llm_boundary_no_core_or_domain_coupling() -> None:
    """Verify ``ares.llm`` imports neither core ``ares.machine`` nor domain/DuckDB modules."""
    forbidden_prefixes = (
        "ares.machine",
        "ares.store",
        "ares.types",
        "ares.cli",
        "duckdb",
        "sql",
        "fastapi",
        "uvicorn",
    )
    llm_dir = Path(__file__).resolve().parent.parent / "ares" / "llm"
    for py_file in llm_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden_prefixes), (
                        f"Forbidden import {alias.name!r} in {py_file.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden_prefixes), (
                    f"Forbidden import {node.module!r} in {py_file.name}"
                )

    probe = (
        "import sys, ares.llm; "
        "bad = [m for m in ('duckdb', 'fastapi', 'uvicorn') if m in sys.modules]; "
        "assert not bad, f'Unexpectedly imported: {bad}'"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_structured_validation_retry_accumulates_usage_and_repairs() -> None:
    """Retry invalid JSON / schema outputs up to ``max_validation_retries`` and accumulate usage."""
    seen_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(body)
        if len(seen_payloads) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": '{"answer": "ok"}'}}],  # missing confidence
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer": "ok", "confidence": 0.95}'}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            },
        )

    client = OpenAICompatClient(
        max_validation_retries=1,
        transport=httpx.MockTransport(handler),
    )
    resp = client.complete(
        [LLMMessage(role="user", content="check")],
        response_model=_StructuredAnswer,
    )
    assert len(seen_payloads) == 2
    assert isinstance(resp.parsed, _StructuredAnswer)
    assert resp.parsed.answer == "ok"
    assert resp.parsed.confidence == pytest.approx(0.95)
    assert resp.usage.prompt_tokens == 30
    assert resp.usage.completion_tokens == 13
    assert resp.usage.total_tokens == 43
    assert "failed JSON/schema validation" in seen_payloads[1]["messages"][-1]["content"]


def test_structured_validation_retries_zero_raises_immediately() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not json at all"}}]},
        )

    client = OpenAICompatClient(
        max_validation_retries=0,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises((json.JSONDecodeError, ValidationError)):
        client.complete(
            [LLMMessage(role="user", content="check")],
            response_model=_StructuredAnswer,
        )
    assert calls == 1
