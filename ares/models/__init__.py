"""Pluggable LLM model provider protocol and adapters for ARES."""

from ares.models.base import (
    ContextLengthExceededError,
    ModelProvider,
    ModelTelemetry,
)
from ares.models.gemini import (
    GeminiModelProvider,
    extract_telemetry,
)
from ares.models.openai_compat import OpenAICompatibleProvider

__all__ = [
    "ContextLengthExceededError",
    "GeminiModelProvider",
    "ModelProvider",
    "ModelTelemetry",
    "OpenAICompatibleProvider",
    "extract_telemetry",
]
