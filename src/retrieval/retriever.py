"""Retriever combining vector search with structured queries."""

from __future__ import annotations

import re

import structlog

from src.embeddings.embedder import Embedder
from src.models.schemas import (
    RetrievalResult,
    SearchResult,
)
from src.storage.structured_store import StructuredStore
from src.storage.vector_store import VectorStore

logger = structlog.get_logger()

# Patterns for extracting speaker names from queries
_SPEAKER_PATTERNS = [
    re.compile(r"what did (\w+) say", re.IGNORECASE),
    re.compile(r"what does (\w+) think", re.IGNORECASE),
    re.compile(r"what were (\w+)'s", re.IGNORECASE),
    re.compile(r"what was (\w+)'s", re.IGNORECASE),
    re.compile(r"(\w+)'s (?:opinion|view|concern|suggestion|comment)", re.IGNORECASE),
    re.compile(r"according to (\w+)", re.IGNORECASE),
    re.compile(r"did (\w+) (?:mention|say|speak|talk|contribute|attend|participate)", re.IGNORECASE),
    re.compile(r"was (\w+) (?:in|at|present|part of)", re.IGNORECASE),
    re.compile(r"tell me about (\w+)", re.IGNORECASE),
    re.compile(r"what (?:did|does|has) (\w+)", re.IGNORECASE),
]


class Retriever:
    """Combines vector similarity search with structured SQLite queries.

    Routes retrieval based on query intent:
    - Semantic/speaker queries use Qdrant vector search with optional filters.
    - Structured queries use SQLite for pre-extracted action items and decisions.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        structured_store: StructuredStore,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._structured_store = structured_store
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    def search(
        self,
        query: str,
        meeting_id: str | None = None,
        speaker_filter: str | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Perform vector similarity search with optional filters.

        Args:
            query: The search query text.
            meeting_id: Optional filter to a specific meeting.
            speaker_filter: Optional filter to a specific speaker.
            top_k: Number of results (defaults to config value).

        Returns:
            RetrievalResult with ranked chunks, with low_confidence flag if scores are poor.
        """
        k = top_k or self._top_k
        query_embedding = self._embedder.embed(query)

        # Speaker-filtered queries: skip similarity threshold entirely.
        # "What did Tom say?" has near-zero semantic overlap with Tom's actual
        # utterances, so threshold-based filtering would drop all results.
        # We want ALL of that speaker's top-K chunks regardless of score.
        effective_threshold = 0.0 if speaker_filter else self._similarity_threshold

        results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            meeting_id=meeting_id,
            speaker=speaker_filter,
            score_threshold=effective_threshold,
        )

        # Check for low confidence results
        low_confidence = False
        if results:
            top_score = results[0].score
            if top_score < 0.5:
                low_confidence = True
                logger.warning(
                    "low_confidence_retrieval",
                    query_preview=query[:80],
                    top_score=round(top_score, 4),
                    num_results=len(results),
                    meeting_id=meeting_id,
                    speaker_filter=speaker_filter,
                    hint="Scores below 0.5 suggest weak semantic match. "
                         "Check embedding normalisation and query phrasing.",
                )
        elif not results:
            logger.warning(
                "retrieval_no_results_above_threshold",
                query_preview=query[:80],
                threshold=effective_threshold,
                meeting_id=meeting_id,
                speaker_filter=speaker_filter,
            )
            # Retry without threshold to get top 3 as a last resort
            results = self._vector_store.search(
                query_embedding=query_embedding,
                top_k=min(k, 3),
                meeting_id=meeting_id,
                speaker=speaker_filter,
                score_threshold=0.0,
            )
            if results:
                low_confidence = True
                logger.warning(
                    "retrieval_fallback_low_confidence",
                    query_preview=query[:80],
                    num_results=len(results),
                    top_score=round(results[0].score, 4) if results else 0,
                )

        logger.info(
            "retrieval_search",
            query_preview=query[:80],
            results=len(results),
            meeting_id=meeting_id,
            speaker_filter=speaker_filter,
            low_confidence=low_confidence,
        )

        return RetrievalResult(chunks=results, low_confidence=low_confidence)

    def get_structured(
        self, query: str, meeting_id: str | None = None
    ) -> RetrievalResult:
        """Query structured store for action items, decisions, and summaries.

        Formats the structured data as a context string suitable for the LLM.

        Args:
            query: The user query (used to decide what to retrieve).
            meeting_id: Optional filter to a specific meeting.

        Returns:
            RetrievalResult with structured_context populated.
        """
        parts: list[str] = []
        query_lower = query.lower()

        # Get speakers/attendees if relevant
        if any(kw in query_lower for kw in ("who", "attend", "speaker", "participant", "present", "member")):
            speakers = self._structured_store.get_speakers(meeting_id)
            if speakers:
                parts.append("MEETING SPEAKERS/ATTENDEES:\n  - " + "\n  - ".join(speakers))

        # Get action items if relevant
        if any(kw in query_lower for kw in ("action", "todo", "task", "assign", "deadline")):
            items = self._structured_store.get_action_items(meeting_id)
            if items:
                lines = ["ACTION ITEMS:"]
                for item in items:
                    if item.deadline:
                        deadline = f" (deadline: {item.deadline})"
                    else:
                        deadline = " (no specific deadline)"
                    lines.append(f"  - {item.assignee}: {item.task}{deadline}")
                parts.append("\n".join(lines))

        # Get decisions if relevant
        if any(kw in query_lower for kw in ("decision", "decided", "agreed", "conclusion")):
            decisions = self._structured_store.get_decisions(meeting_id)
            if decisions:
                lines = ["DECISIONS:"]
                for dec in decisions:
                    by = ", ".join(dec.decided_by) if dec.decided_by else "team"
                    lines.append(f"  - {dec.decision} (by {by}): {dec.context}")
                parts.append("\n".join(lines))

        # Get summary if it's a summary-type query
        if any(kw in query_lower for kw in ("summar", "overview", "recap", "tldr")):
            if meeting_id:
                summary = self._structured_store.get_meeting_summary(meeting_id)
                if summary:
                    parts.append(f"MEETING SUMMARY:\n{summary}")
            else:
                # Summaries across all meetings
                meetings = self._structured_store.list_meetings()
                for m in meetings:
                    summary = self._structured_store.get_meeting_summary(m.meeting_id)
                    if summary:
                        parts.append(f"MEETING {m.meeting_id} SUMMARY:\n{summary}")

        # If no structured data matched, fall back to getting everything
        if not parts:
            items = self._structured_store.get_action_items(meeting_id)
            decisions = self._structured_store.get_decisions(meeting_id)
            if items:
                lines = ["ACTION ITEMS:"]
                for item in items:
                    if item.deadline:
                        deadline = f" (deadline: {item.deadline})"
                    else:
                        deadline = " (no specific deadline)"
                    lines.append(f"  - {item.assignee}: {item.task}{deadline}")
                parts.append("\n".join(lines))
            if decisions:
                lines = ["DECISIONS:"]
                for dec in decisions:
                    by = ", ".join(dec.decided_by) if dec.decided_by else "team"
                    lines.append(f"  - {dec.decision} (by {by}): {dec.context}")
                parts.append("\n".join(lines))

        structured_context = "\n\n".join(parts) if parts else "No structured data available."

        # Only supplement with vector search if structured data is insufficient
        chunks: list[SearchResult] = []
        if not parts:
            # No structured data matched — fall back to vector search
            vector_results = self.search(query, meeting_id)
            chunks = vector_results.chunks
            logger.warning(
                "structured_retrieval_no_structured_data",
                query_preview=query[:80],
                falling_back_to_vector=True,
                vector_results=len(chunks),
            )
        else:
            logger.info(
                "structured_retrieval",
                query_preview=query[:80],
                has_structured=True,
                skipped_vector_search=True,
            )

        return RetrievalResult(
            chunks=chunks,
            structured_context=structured_context,
        )

    def get_known_speakers(self, meeting_id: str | None = None) -> set[str]:
        """Get all known speaker names from ingested meetings.

        Args:
            meeting_id: If provided, only speakers from this meeting.

        Returns:
            Set of speaker names.
        """
        meetings = self._structured_store.list_meetings()
        speakers: set[str] = set()
        for m in meetings:
            if meeting_id is None or m.meeting_id == meeting_id:
                speakers.update(m.speakers)
        return speakers

    @staticmethod
    def extract_speaker(query: str) -> str | None:
        """Extract a speaker name from a query using regex patterns.

        Args:
            query: The user query.

        Returns:
            Extracted speaker name (title-cased) or None.
        """
        for pattern in _SPEAKER_PATTERNS:
            match = pattern.search(query)
            if match:
                name = match.group(1).strip().title()
                # Filter out common false positives
                if name.lower() not in ("the", "this", "that", "what", "how", "why", "who", "it"):
                    return name
        return None
