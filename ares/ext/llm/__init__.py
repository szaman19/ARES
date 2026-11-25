"""Optional LLM adapter package for ARES."""

from ares.ext.llm.base import (
    ContextLengthExceededError,
    LLMClient,
    LLMMessage,
    LLMResponse,
    LLMUsage,
)
from ares.ext.llm.openai_compat import OpenAICompatClient

__all__ = [
    "ContextLengthExceededError",
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "LLMUsage",
    "OpenAICompatClient",
]
