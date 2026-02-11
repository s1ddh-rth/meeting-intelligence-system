"""Embedding wrapper around sentence-transformers with lazy loading."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()

# Module-level singleton for the model instance
_model_instance: SentenceTransformer | None = None


class Embedder:
    """Generates embeddings using sentence-transformers.

    The model is loaded lazily on first use and cached as a singleton
    to avoid repeated loading in multi-request scenarios.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load and cache the sentence-transformer model."""
        global _model_instance
        if _model_instance is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading_embedding_model", model=self.model_name)
            start = time.time()
            _model_instance = SentenceTransformer(self.model_name)
            elapsed = (time.time() - start) * 1000
            logger.info("embedding_model_loaded", model=self.model_name, load_time_ms=round(elapsed))
        return _model_instance

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        model = self._get_model()
        vector = model.encode(text, show_progress_bar=False)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts efficiently.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        model = self._get_model()
        start = time.time()
        vectors = model.encode(texts, show_progress_bar=False, batch_size=32)
        elapsed = (time.time() - start) * 1000

        logger.info(
            "batch_embedding_complete",
            num_texts=len(texts),
            latency_ms=round(elapsed),
        )
        return [v.tolist() for v in vectors]
