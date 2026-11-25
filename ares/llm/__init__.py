"""Optional LLM adapter package for ARES."""

from ares.llm.base import (
    ContextLengthExceededError,
    LLMClient,
    LLMMessage,
    LLMResponse,
    LLMUsage,
)
from ares.llm.openai_compat import OpenAICompatClient

__all__ = [
    "ContextLengthExceededError",
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "LLMUsage",
    "OpenAICompatClient",
]
