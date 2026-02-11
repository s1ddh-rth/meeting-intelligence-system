"""Factory for creating LLM provider instances based on configuration."""

from __future__ import annotations

import structlog

from src.config.settings import Settings
from src.llm.provider import LLMProvider

logger = structlog.get_logger()


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Create an LLM provider based on the configured provider name.

    Args:
        settings: Application settings.

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If the configured provider is unknown.
    """
    match settings.llm_provider:
        case "gemini":
            from src.llm.gemini_provider import GeminiProvider

            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required when llm_provider='gemini'")
            return GeminiProvider(api_key=settings.gemini_api_key)

        case "anthropic":
            from src.llm.anthropic_provider import AnthropicProvider

            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required when llm_provider='anthropic'")
            return AnthropicProvider(api_key=settings.anthropic_api_key)

        case "ollama":
            from src.llm.ollama_provider import OllamaProvider

            return OllamaProvider(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
            )

        case _:
            raise ValueError(
                f"Unknown LLM provider: {settings.llm_provider}. "
                "Supported: 'gemini', 'anthropic', 'ollama'"
            )
