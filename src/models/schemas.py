"""Pydantic models for all data structures in the Meeting Intelligence System."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


# --- Transcript Parsing ---


class Utterance(BaseModel):
    """Single spoken turn from a meeting transcript."""

    speaker: str
    timestamp: str | None = None
    text: str


# --- Chunking ---


class ChunkMetadata(BaseModel):
    """Metadata attached to every transcript chunk for filtering and tracing."""

    meeting_id: str
    speaker: str
    timestamp_start: str | None = None
    timestamp_end: str | None = None
    chunk_index: int
    num_tokens: int


class Chunk(BaseModel):
    """A chunk of transcript text with metadata, ready for embedding."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    metadata: ChunkMetadata


# --- Query Types ---


class QueryIntent(str, Enum):
    """Classification of user query intent for routing."""

    STRUCTURED = "structured"
    SPEAKER_SPECIFIC = "speaker"
    SEMANTIC = "semantic"
    CROSS_MEETING = "cross_meeting"


class ChunkSource(BaseModel):
    """A retrieved chunk returned as a source citation."""

    text: str
    speaker: str
    timestamp: str | None = None
    meeting_id: str
    relevance_score: float


class ConfidenceLevel(str, Enum):
    """Answer confidence based on retrieval quality."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QueryResponse(BaseModel):
    """Response to a user query with answer, sources, and metrics."""

    answer: str
    sources: list[ChunkSource]
    intent: QueryIntent
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_score: float = 0.0
    tokens_used: int | None = None
    latency_ms: float


# --- Structured Extractions ---


class ActionItem(BaseModel):
    """An action item extracted from a meeting."""

    assignee: str
    task: str
    deadline: str | None = None


class Decision(BaseModel):
    """A decision made during a meeting."""

    decision: str
    context: str
    decided_by: list[str] = Field(default_factory=list)


class MeetingExtractions(BaseModel):
    """All structured data extracted from a single meeting."""

    meeting_id: str
    action_items: list[ActionItem] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    summary: str = ""
    speakers: list[str] = Field(default_factory=list)


class MeetingInfo(BaseModel):
    """Summary info about an ingested meeting."""

    meeting_id: str
    filename: str
    speakers: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    num_chunks: int = 0
    ingested_at: datetime | None = None


# --- Ingestion ---


class IngestionResult(BaseModel):
    """Result returned after ingesting a meeting transcript."""

    meeting_id: str
    chunks_created: int
    speakers: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    action_items_count: int = 0


class AudioIngestionResult(IngestionResult):
    """Result returned after ingesting an audio file (transcription + ingestion)."""

    transcript_filename: str


# --- LLM ---


class LLMResponse(BaseModel):
    """Response from an LLM provider."""

    content: str
    model: str
    tokens_used: int | None = None


# --- Retrieval ---


class SearchResult(BaseModel):
    """A single vector search result from Qdrant."""

    chunk_id: str
    text: str
    score: float
    metadata: ChunkMetadata


class RetrievalResult(BaseModel):
    """Combined retrieval result with chunks and optional structured data."""

    chunks: list[SearchResult] = Field(default_factory=list)
    structured_context: str | None = None
    low_confidence: bool = False

    def to_sources(self) -> list[ChunkSource]:
        """Convert search results to source citations."""
        return [
            ChunkSource(
                text=chunk.text,
                speaker=chunk.metadata.speaker,
                timestamp=chunk.metadata.timestamp_start,
                meeting_id=chunk.metadata.meeting_id,
                relevance_score=chunk.score,
            )
            for chunk in self.chunks
        ]

    def format_context(self) -> str:
        """Format retrieved chunks as context string for LLM prompt."""
        parts: list[str] = []
        if self.low_confidence and self.chunks:
            parts.append(
                "NOTE: The following transcript excerpts have low relevance scores "
                "and may not closely match the query. Use them cautiously and "
                "indicate if the information may not fully address the question."
            )
        if self.structured_context:
            parts.append(self.structured_context)
        for chunk in self.chunks:
            speaker = chunk.metadata.speaker
            ts = chunk.metadata.timestamp_start or "N/A"
            meeting = chunk.metadata.meeting_id
            parts.append(
                f"[Speaker: {speaker}, Time: {ts}, Meeting: {meeting}]\n{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)
