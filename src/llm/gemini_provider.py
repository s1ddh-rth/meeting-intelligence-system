"""Google Gemini LLM provider implementation."""

from __future__ import annotations

import asyncio
import time

import structlog
from google import genai
from google.genai import types

from src.llm.provider import LLMProvider, RateLimitError
from src.models.schemas import LLMResponse

logger = structlog.get_logger()

# Gemini free tier: 10 RPM — retry config
_MAX_RETRIES = 3
_BASE_DELAY = 2.0


class GeminiProvider(LLMProvider):
    """Google Gemini 2.5 Flash provider via the google-genai SDK.

    Handles rate limiting with exponential backoff for the free tier (10 RPM).
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        logger.info("gemini_provider_initialized", model=self._model)

    async def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate a response using Gemini with retry logic.

        Args:
            system_prompt: System instructions.
            user_prompt: User query.

        Returns:
            LLMResponse with generated content.
        """
        start = time.time()
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self._model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.3,
                    ),
                )

                elapsed = (time.time() - start) * 1000
                content = response.text or ""
                tokens_used = None
                if response.usage_metadata:
                    tokens_used = (
                        (response.usage_metadata.prompt_token_count or 0)
                        + (response.usage_metadata.candidates_token_count or 0)
                    )

                logger.info(
                    "llm_call",
                    model=self._model,
                    tokens_used=tokens_used,
                    latency_ms=round(elapsed),
                    attempt=attempt + 1,
                )

                return LLMResponse(
                    content=content,
                    model=self._model,
                    tokens_used=tokens_used,
                )

            except Exception as e:
                last_error = e
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "gemini_call_failed",
                    attempt=attempt + 1,
                    error=str(e),
                    retry_delay=delay,
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(delay)

        logger.error("gemini_call_exhausted_retries", error=str(last_error))
        error_str = str(last_error)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            raise RateLimitError(
                f"Gemini API rate limit exceeded (free tier: 20 requests/day). "
                f"Please wait a minute and try again.",
            )
        raise RuntimeError(f"Gemini API call failed after {_MAX_RETRIES} retries: {last_error}")
