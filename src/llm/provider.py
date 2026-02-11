"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.schemas import LLMResponse


class RateLimitError(Exception):
    """Raised when the LLM provider hits a rate limit (e.g. 429)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMProvider(ABC):
    """Abstract interface for language model providers.

    All LLM implementations (Gemini, Anthropic, Ollama) must implement
    this interface, enabling provider-agnostic orchestration.
    """

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: The user's query or input.

        Returns:
            LLMResponse with content, model name, and token usage.
        """
        ...
