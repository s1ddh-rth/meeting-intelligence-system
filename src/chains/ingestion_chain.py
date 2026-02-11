"""Orchestrates the full meeting transcript ingestion pipeline."""

from __future__ import annotations

import re

import structlog

from src.embeddings.embedder import Embedder
from src.ingestion.chunker import SpeakerAwareChunker
from src.ingestion.extractor import StructuredExtractor
from src.ingestion.parser import TranscriptParser
from src.models.schemas import IngestionResult
from src.storage.structured_store import StructuredStore
from src.storage.vector_store import VectorStore

logger = structlog.get_logger()


class IngestionChain:
    """Orchestrates: parse -> chunk -> embed -> store vectors -> extract -> store structured.

    Each step is handled by a dedicated component, wired together via dependency injection.
    """

    def __init__(
        self,
        parser: TranscriptParser,
        chunker: SpeakerAwareChunker,
        embedder: Embedder,
        vector_store: VectorStore,
        structured_store: StructuredStore,
        extractor: StructuredExtractor,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._structured_store = structured_store
        self._extractor = extractor

    async def run(self, file_content: str, filename: str) -> IngestionResult:
        """Run the full ingestion pipeline for a transcript.

        Args:
            file_content: Raw transcript text.
            filename: Original file name (used to derive meeting_id).

        Returns:
            IngestionResult with stats about the ingestion.
        """
        meeting_id = self._derive_meeting_id(filename)
        logger.info("ingestion_started", meeting_id=meeting_id, filename=filename)

        # Step 1: Parse transcript into utterances
        utterances = self._parser.parse(file_content)
        if not utterances:
            logger.warning("ingestion_empty_transcript", meeting_id=meeting_id)
            return IngestionResult(meeting_id=meeting_id, chunks_created=0)

        # Step 2: Chunk utterances with speaker awareness
        chunks = self._chunker.chunk(utterances, meeting_id)

        # Step 3: Generate embeddings for all chunks
        embeddings = self._embedder.embed_batch([c.text for c in chunks])

        # Step 4: Delete existing data for this meeting (idempotent re-ingestion)
        try:
            self._vector_store.delete_meeting(meeting_id)
        except Exception:
            logger.warning("delete_existing_vectors_failed", meeting_id=meeting_id)

        # Step 5: Store vectors in Qdrant
        self._vector_store.upsert_chunks(chunks, embeddings)

        # Step 6: Extract structured data via LLM
        extractions = await self._extractor.extract(utterances, meeting_id)

        # Step 7: Store structured data in SQLite
        self._structured_store.store_extractions(
            extractions, filename=filename, num_chunks=len(chunks)
        )

        logger.info(
            "ingestion_complete",
            meeting_id=meeting_id,
            chunks=len(chunks),
            speakers=extractions.speakers,
            action_items=len(extractions.action_items),
            decisions=len(extractions.decisions),
        )

        return IngestionResult(
            meeting_id=meeting_id,
            chunks_created=len(chunks),
            speakers=extractions.speakers,
            topics=extractions.topics,
            action_items_count=len(extractions.action_items),
        )

    @staticmethod
    def _derive_meeting_id(filename: str) -> str:
        """Derive a stable meeting ID from the filename.

        Strips extension and normalises to a URL-safe identifier.
        Examples:
            'standup_2025_01_15.txt' -> 'standup_2025_01_15'
            'Sprint Planning 2025-01-20.txt' -> 'sprint_planning_2025-01-20'
        """
        # Remove extension
        name = re.sub(r"\.[^.]+$", "", filename)
        # Replace spaces and special chars with underscores
        name = re.sub(r"[^\w\-]", "_", name)
        # Collapse multiple underscores
        name = re.sub(r"_+", "_", name).strip("_")
        return name.lower()
