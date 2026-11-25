"""OpenAI-compatible HTTP chat-completions adapter using ``httpx``."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from ares.llm.base import (
    ContextLengthExceededError,
    LLMMessage,
    LLMResponse,
    LLMUsage,
)

_CONTEXT_LENGTH_MARKERS: tuple[str, ...] = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "too many tokens",
    "token limit",
    "prompt is too long",
    "exceeds the available context",
    "context window",
)

_RESPONSE_FORMAT_UNSUPPORTED_MARKERS: tuple[str, ...] = (
    "response_format",
    "json_schema",
    "unsupported",
)

_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)


def _strip_json_fences(text: str) -> str:
    """Strip surrounding markdown code fences from JSON text if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        inner = stripped[first_nl + 1 :] if first_nl != -1 else stripped[3:]
        inner_rstrip = inner.rstrip()
        if inner_rstrip.endswith("```"):
            inner_rstrip = inner_rstrip[:-3]
        return inner_rstrip.strip()

    match = _FENCE_RE.search(stripped)
    if match and not (stripped.startswith("{") or stripped.startswith("[")):
        return match.group(1).strip()
    return stripped


def _is_context_length_error(status_code: int, body_text: str) -> bool:
    """Return True when an HTTP error indicates prompt/context token overflow."""
    if status_code not in (400, 413, 422):
        return False
    lowered = body_text.lower()
    return any(marker in lowered for marker in _CONTEXT_LENGTH_MARKERS)


def _is_response_format_rejected(status_code: int, body_text: str) -> bool:
    """Return True when the server rejects ``response_format`` / ``json_schema``."""
    if status_code not in (400, 422):
        return False
    lowered = body_text.lower()
    return any(marker in lowered for marker in _RESPONSE_FORMAT_UNSUPPORTED_MARKERS)


def _append_schema_instructions(
    messages: list[dict[str, str]],
    response_model: type[BaseModel],
) -> list[dict[str, str]]:
    """Return a copy of ``messages`` with concise JSON schema instructions appended."""
    schema_json = json.dumps(response_model.model_json_schema(), separators=(",", ":"))
    instruction = (
        f"Respond ONLY with valid JSON matching the {response_model.__name__} schema "
        f"(no markdown fences or commentary): {schema_json}"
    )
    updated = [dict(m) for m in messages]
    for idx in range(len(updated) - 1, -1, -1):
        if updated[idx]["role"] == "user":
            updated[idx]["content"] = f"{updated[idx]['content']}\n\n{instruction}"
            return updated
    if updated:
        updated[-1]["content"] = f"{updated[-1]['content']}\n\n{instruction}"
        return updated
    updated.append({"role": "user", "content": instruction})
    return updated


def _extract_text(data: dict[str, Any]) -> str:
    """Extract assistant text content from an OpenAI chat completion payload."""
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        return ""
    return str(content)


def _extract_usage(data: dict[str, Any], latency_ms: float) -> LLMUsage:
    """Extract token counts and measured latency into ``LLMUsage``."""
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=round(latency_ms, 3),
    )


class OpenAICompatClient:
    """OpenAI-compatible ``/chat/completions`` client built directly on ``httpx``."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:3b",
        api_key: str | None = None,
        timeout_s: float = 60.0,
        reasoning_effort: str | None = None,
        max_validation_retries: int = 1,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort
        self.max_validation_retries = max(0, int(max_validation_retries))
        self.transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_messages(
        self,
        messages: Sequence[LLMMessage],
        system: str | None,
    ) -> list[dict[str, str]]:
        serialized: list[dict[str, str]] = []
        if system is not None:
            serialized.append({"role": "system", "content": system})
        for msg in messages:
            serialized.append({"role": msg.role, "content": msg.content})
        return serialized

    def _post(
        self,
        client: httpx.Client,
        payload: dict[str, Any],
    ) -> httpx.Response:
        return client.post(
            self.endpoint,
            headers=self._headers(),
            json=payload,
        )

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat-completion request and return a normalized ``LLMResponse``."""
        serialized_messages = self._build_messages(messages, system)
        use_response_format = response_model is not None

        def _make_payload(msgs: list[dict[str, str]], *, with_schema_fmt: bool) -> dict[str, Any]:
            body: dict[str, Any] = {
                "model": self.model,
                "messages": msgs,
                "temperature": temperature,
            }
            if max_tokens is not None:
                body["max_tokens"] = max_tokens
            if self.reasoning_effort is not None:
                body["reasoning_effort"] = self.reasoning_effort
            if response_model is not None and with_schema_fmt:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": response_model.model_json_schema(),
                        "strict": True,
                    },
                }
            return body

        total_usage = LLMUsage()
        current_messages = list(serialized_messages)
        max_attempts = 1 + (self.max_validation_retries if response_model is not None else 0)

        with httpx.Client(timeout=self.timeout_s, transport=self.transport) as client:
            for attempt in range(max_attempts):
                t0 = time.perf_counter()
                req_messages = (
                    current_messages
                    if (use_response_format or response_model is None)
                    else _append_schema_instructions(current_messages, response_model)
                )
                payload = _make_payload(req_messages, with_schema_fmt=use_response_format)
                resp = self._post(client, payload)

                if _is_context_length_error(resp.status_code, resp.text):
                    raise ContextLengthExceededError(
                        f"Context length exceeded (HTTP {resp.status_code}): {resp.text}"
                    )

                if (
                    response_model is not None
                    and use_response_format
                    and _is_response_format_rejected(resp.status_code, resp.text)
                ):
                    use_response_format = False
                    fallback_payload = _make_payload(
                        _append_schema_instructions(current_messages, response_model),
                        with_schema_fmt=False,
                    )
                    resp = self._post(client, fallback_payload)
                    if _is_context_length_error(resp.status_code, resp.text):
                        raise ContextLengthExceededError(
                            f"Context length exceeded (HTTP {resp.status_code}): {resp.text}"
                        )

                resp.raise_for_status()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                data: dict[str, Any] = resp.json()
                step_usage = _extract_usage(data, latency_ms)
                total_usage = LLMUsage(
                    prompt_tokens=total_usage.prompt_tokens + step_usage.prompt_tokens,
                    completion_tokens=total_usage.completion_tokens + step_usage.completion_tokens,
                    total_tokens=total_usage.total_tokens + step_usage.total_tokens,
                    latency_ms=round(total_usage.latency_ms + step_usage.latency_ms, 3),
                )
                text = _extract_text(data)

                if response_model is None:
                    return LLMResponse(
                        text=text,
                        parsed=None,
                        usage=total_usage,
                        raw=data,
                    )

                cleaned = _strip_json_fences(text)
                try:
                    parsed = response_model.model_validate(json.loads(cleaned))
                    return LLMResponse(
                        text=text,
                        parsed=parsed,
                        usage=total_usage,
                        raw=data,
                    )
                except (json.JSONDecodeError, ValidationError) as exc:
                    if attempt + 1 >= max_attempts:
                        raise
                    schema_json = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
                    current_messages = [
                        *current_messages,
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": (
                                f"Your previous output failed JSON/schema validation ({exc}). "
                                f"Return ONLY valid JSON matching schema: {schema_json}"
                            ),
                        },
                    ]

        raise RuntimeError("Unreachable OpenAICompatClient completion loop exit")
