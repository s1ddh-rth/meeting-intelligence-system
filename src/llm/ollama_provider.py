"""Ollama local LLM provider implementation."""

from __future__ import annotations

import time

import httpx
import structlog

from src.llm.provider import LLMProvider
from src.models.schemas import LLMResponse

logger = structlog.get_logger()


class OllamaProvider(LLMProvider):
    """Local Ollama provider via HTTP API.

    Fallback provider for fully offline operation. Requires a running
    Ollama server with the configured model pulled.
    """

    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "llama3.2"
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        logger.info("ollama_provider_initialized", model=self._model, base_url=self._base_url)

    async def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate a response using a local Ollama model.

        Args:
            system_prompt: System instructions.
            user_prompt: User query.

        Returns:
            LLMResponse with generated content.
        """
        start = time.time()

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()

        elapsed = (time.time() - start) * 1000
        content = data.get("message", {}).get("content", "")
        tokens_used = data.get("eval_count")

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
