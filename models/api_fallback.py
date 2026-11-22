"""Multi-key Gemini + cross-provider (OpenAI / Anthropic / OpenRouter) vision fallback helper."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx


def get_gemini_api_keys(primary: str | None = None) -> list[str]:
    """Collect unique non-empty Gemini API keys in priority order."""
    candidates: list[str] = []
    if primary:
        candidates.append(primary)
    for env_name in (
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_FALLBACK",
        "GEMINI_API_KEY_2",
        "GEMINI_API_KEY_3",
        "GOOGLE_API_KEY",
    ):
        val = os.getenv(env_name, "").strip()
        if val:
            candidates.append(val)
    multi = os.getenv("GEMINI_API_KEYS", "").strip()
    if multi:
        for part in multi.split(","):
            p = part.strip()
            if p:
                candidates.append(p)

    seen: set[str] = set()
    ordered: list[str] = []
    for key in candidates:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def get_fallback_vision_providers() -> list[tuple[str, str, str]]:
    """Return configured non-Gemini vision fallback providers as (provider, api_key, model)."""
    providers: list[tuple[str, str, str]] = []
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        providers.append(("openai", openai_key, os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")))
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        providers.append(("anthropic", anthropic_key, os.getenv("ANTHROPIC_VISION_MODEL", "claude-haiku-4-5")))
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        providers.append(("openrouter", openrouter_key, os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini")))
    return providers


def has_any_vision_api_key(primary: str | None = None) -> bool:
    """Return True if at least one Gemini key or fallback vision provider key is configured."""
    return bool(get_gemini_api_keys(primary) or get_fallback_vision_providers())


def call_fallback_vision_json(
    provider: str,
    api_key: str,
    model: str,
    *,
    prompt: str,
    system_instruction: str,
    b64_img: str,
    schema: dict[str, Any],
    mime_type: str = "image/jpeg",
    timeout_s: float = 30.0,
) -> tuple[str, int, int, float]:
    """Call OpenAI, Anthropic, or OpenRouter vision endpoint and return (json_text, prompt_toks, completion_toks, latency_ms)."""
    schema_hint = (
        f"{prompt}\n\nReturn ONLY valid JSON matching this JSON schema:\n"
        f"{json.dumps(schema)}"
    )
    t0 = time.perf_counter()

    if provider in ("openai", "openrouter"):
        url = (
            "https://api.openai.com/v1/chat/completions"
            if provider == "openai"
            else "https://openrouter.ai/api/v1/chat/completions"
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": schema_hint},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_img}"},
                        },
                    ],
                },
            ],
        }
        resp = httpx.post(url, headers=headers, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        return (
            text or "{}",
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            latency_ms,
        )

    if provider == "anthropic":
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 512,
            "system": system_instruction,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": b64_img,
                            },
                        },
                        {"type": "text", "text": schema_hint},
                    ],
                }
            ],
        }
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        usage = data.get("usage") or {}
        return (
            text or "{}",
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
            latency_ms,
        )

    raise ValueError(f"Unsupported fallback vision provider: {provider}")
