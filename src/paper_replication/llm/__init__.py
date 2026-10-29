"""LLM provider adapters: Gemini (default for real runs), OpenAI (optional) and a deterministic mock."""

from __future__ import annotations

from ..config import Settings, settings_from_env
from .base import (
    LLMAuthError,
    LLMError,
    LLMOutputError,
    LLMProvider,
    LLMRateLimitError,
    LLMRequest,
    LLMRequestError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    ModelNotAvailableError,
)
from .mock_provider import MockProvider


def get_provider(settings: Settings | None = None) -> LLMProvider:
    """Build the provider named in ``settings``. No fallback: invalid/missing configuration raises ConfigError."""
    s = settings or settings_from_env()
    s.validate()
    if s.provider == "mock":
        return MockProvider(s.model or "mock-deterministic")
    if s.provider == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider(s.model, s.gemini_api_key, s.timeout_s, s.max_retries, s.temperature)
    from .openai_provider import OpenAIProvider

    return OpenAIProvider(s.model, s.openai_api_key, s.timeout_s, s.max_retries, s.temperature)


__all__ = [
    "LLMAuthError", "LLMError", "LLMOutputError", "LLMProvider", "LLMRateLimitError", "LLMRequest", "LLMRequestError",
    "LLMResponse", "LLMTimeoutError", "LLMUnavailableError", "MockProvider", "ModelNotAvailableError", "get_provider",
]
