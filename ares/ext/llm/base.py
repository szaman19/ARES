"""Provider-agnostic LLM client contracts and message/response models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """Single chat message sent to an LLM provider."""

    role: Literal["system", "user", "assistant"]
    content: str


class LLMUsage(BaseModel):
    """Token accounting and timing metadata for a single completion call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


class LLMResponse(BaseModel):
    """Normalized response returned by an ``LLMClient`` implementation."""

    text: str
    parsed: Any = None
    reasoning: str | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)
    raw: dict[str, Any] = Field(default_factory=dict)


class ContextLengthExceededError(RuntimeError):
    """Raised when an LLM provider rejects a request due to context/token limits."""


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for synchronous chat-completion LLM adapters."""

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Execute a chat completion and return a normalized ``LLMResponse``."""
        ...
