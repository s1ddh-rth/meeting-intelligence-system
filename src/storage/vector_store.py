"""Qdrant vector store wrapper for meeting transcript chunks."""

from __future__ import annotations

import time

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.models.schemas import Chunk, ChunkMetadata, SearchResult

logger = structlog.get_logger()


class VectorStore:
    """Manages Qdrant operations for meeting transcript vectors.

    Provides collection initialization, chunk upserting, similarity search
    with metadata filtering, and meeting deletion.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "meetings",
        embedding_dimension: int = 384,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self._client = QdrantClient(host=host, port=port, timeout=30)
        logger.info("qdrant_client_created", host=host, port=port)

    def init_collection(self) -> None:
        """Create the collection if it does not exist."""
        try:
            collections = self._client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "collection_created",
                    name=self.collection_name,
                    dimension=self.embedding_dimension,
                )
            else:
                logger.info("collection_exists", name=self.collection_name)
        except Exception:
            logger.exception("collection_init_failed")
            raise

    def upsert_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Batch upsert chunks with their embeddings into Qdrant.

        Args:
            chunks: List of transcript chunks with metadata.
            embeddings: Corresponding embedding vectors.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
            )

        points = [
            PointStruct(
                id=chunk.id,
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "meeting_id": chunk.metadata.meeting_id,
                    "speaker": chunk.metadata.speaker,
                    "timestamp_start": chunk.metadata.timestamp_start,
                    "timestamp_end": chunk.metadata.timestamp_end,
                    "chunk_index": chunk.metadata.chunk_index,
                    "num_tokens": chunk.metadata.num_tokens,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        start = time.time()
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        elapsed = (time.time() - start) * 1000

        logger.info(
            "chunks_upserted",
            count=len(points),
            collection=self.collection_name,
            latency_ms=round(elapsed),
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        meeting_id: str | None = None,
        speaker: str | None = None,
        score_threshold: float = 0.3,
    ) -> list[SearchResult]:
        """Similarity search with optional metadata filters.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            meeting_id: Optional filter by meeting.
            speaker: Optional filter by speaker name.
            score_threshold: Minimum similarity score.

        Returns:
            Ranked list of SearchResult objects.
        """
        conditions: list[FieldCondition] = []
        if meeting_id:
            conditions.append(
                FieldCondition(key="meeting_id", match=MatchValue(value=meeting_id))
            )
        if speaker:
            conditions.append(
                FieldCondition(key="speaker", match=MatchValue(value=speaker))
            )

        query_filter = Filter(must=conditions) if conditions else None

        start = time.time()
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        elapsed = (time.time() - start) * 1000

        search_results = [
            SearchResult(
                chunk_id=str(hit.id),
                text=hit.payload.get("text", ""),
                score=hit.score,
                metadata=ChunkMetadata(
                    meeting_id=hit.payload.get("meeting_id", ""),
                    speaker=hit.payload.get("speaker", ""),
                    timestamp_start=hit.payload.get("timestamp_start"),
                    timestamp_end=hit.payload.get("timestamp_end"),
                    chunk_index=hit.payload.get("chunk_index", 0),
                    num_tokens=hit.payload.get("num_tokens", 0),
                ),
            )
            for hit in response.points
        ]

        logger.info(
            "vector_search_complete",
            results=len(search_results),
            top_k=top_k,
            meeting_id=meeting_id,
            speaker=speaker,
            latency_ms=round(elapsed),
        )
        return search_results

    def delete_meeting(self, meeting_id: str) -> None:
        """Remove all chunks for a given meeting (for re-ingestion).

        Args:
            meeting_id: The meeting to delete.
        """
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="meeting_id", match=MatchValue(value=meeting_id)
                    )
                ]
            ),
        )
        logger.info("meeting_deleted_from_vectors", meeting_id=meeting_id)

    def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            self._client.get_collections()
            return True
        except Exception:
            logger.exception("qdrant_health_check_failed")
            return False
