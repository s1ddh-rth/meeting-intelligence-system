"""Anthropic Claude LLM provider implementation."""

from __future__ import annotations

import time

import structlog

from src.llm.provider import LLMProvider
from src.models.schemas import LLMResponse

logger = structlog.get_logger()


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider via the anthropic SDK.

    Alternative provider for higher-quality responses.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            )
        self._model = model
        logger.info("anthropic_provider_initialized", model=self._model)

    async def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate a response using Claude.

        Args:
            system_prompt: System instructions.
            user_prompt: User query.

        Returns:
            LLMResponse with generated content.
        """
        start = time.time()

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        elapsed = (time.time() - start) * 1000
        content = response.content[0].text if response.content else ""
        tokens_used = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)

        logger.info(
            "llm_call",
            model=self._model,
            tokens_used=tokens_used,
            latency_ms=round(elapsed),
        )

        return LLMResponse(
            content=content,
            model=self._model,
            tokens_used=tokens_used,
        )
