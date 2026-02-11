"""Orchestrates the full query pipeline: classify -> retrieve -> generate."""

from __future__ import annotations

import time

import structlog

from src.llm.provider import LLMProvider, RateLimitError
from src.models.schemas import ConfidenceLevel, QueryIntent, QueryResponse, RetrievalResult
from src.prompts.classification_prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_USER_PROMPT,
)
from src.prompts.query_prompts import QUERY_SYSTEM_PROMPT, QUERY_USER_PROMPT
from src.retrieval.retriever import Retriever

logger = structlog.get_logger()

# Mapping from LLM classification output to QueryIntent enum
_INTENT_MAP = {
    "STRUCTURED": QueryIntent.STRUCTURED,
    "SPEAKER": QueryIntent.SPEAKER_SPECIFIC,
    "SEMANTIC": QueryIntent.SEMANTIC,
    "CROSS_MEETING": QueryIntent.CROSS_MEETING,
}


class QueryChain:
    """Orchestrates: classify intent -> retrieve context -> generate grounded answer.

    Routes queries to the appropriate retrieval strategy based on intent:
    - STRUCTURED: queries SQLite for action items, decisions, summaries.
    - SPEAKER: filters vector search to a specific speaker.
    - SEMANTIC: general vector similarity search.
    - CROSS_MEETING: searches across all meetings.
    """

    def __init__(self, retriever: Retriever, llm: LLMProvider) -> None:
        self._retriever = retriever
        self._llm = llm

    async def run(
        self, query: str, meeting_id: str | None = None
    ) -> QueryResponse:
        """Process a user query end-to-end.

        Args:
            query: The user's question.
            meeting_id: Optional meeting scope.

        Returns:
            QueryResponse with answer, sources, intent, and metrics.
        """
        start_time = time.time()

        # Step 1: Classify intent
        intent = await self._classify_intent(query)

        # Step 2: Retrieve based on intent
        speaker: str | None = None
        match intent:
            case QueryIntent.STRUCTURED:
                context = self._retriever.get_structured(query, meeting_id)

            case QueryIntent.SPEAKER_SPECIFIC:
                speaker = Retriever.extract_speaker(query)

                # If regex didn't find a speaker, try matching against known speakers
                if not speaker:
                    speaker = self._match_known_speaker(query, meeting_id)

                if speaker:
                    # Fast-path: check if this speaker exists before any vector search.
                    # This handles rate-limited LLM scenarios correctly and avoids
                    # the unfiltered-retry fallback returning irrelevant chunks.
                    known_speakers = self._retriever.get_known_speakers(meeting_id)
                    if not self._speaker_exists(speaker, known_speakers):
                        latency = (time.time() - start_time) * 1000
                        answer = self._build_speaker_not_found_answer(
                            speaker, known_speakers, meeting_id
                        )
                        logger.info(
                            "speaker_not_in_known_list",
                            speaker=speaker,
                            known_speakers=sorted(known_speakers),
                            meeting_id=meeting_id,
                        )
                        return QueryResponse(
                            answer=answer,
                            sources=[],
                            intent=intent,
                            confidence=ConfidenceLevel.HIGH,
                            confidence_score=1.0,
                            tokens_used=0,
                            latency_ms=round(latency, 1),
                        )

                    context = self._retriever.search(
                        query, meeting_id, speaker_filter=speaker
                    )
                    # If speaker-filtered search returns nothing, the speaker
                    # may still exist — search without filter to find mentions
                    # of this speaker in other people's utterances
                    if not context.chunks:
                        logger.warning(
                            "speaker_filter_empty_retrying_unfiltered",
                            speaker=speaker,
                            meeting_id=meeting_id,
                        )
                        context = self._retriever.search(query, meeting_id)
                else:
                    # No speaker identified — fall back to semantic search
                    context = self._retriever.search(query, meeting_id)

            case QueryIntent.CROSS_MEETING:
                # Search across all meetings (no meeting_id filter)
                context = self._retriever.search(query, meeting_id=None)

            case _:
                # SEMANTIC — default vector search
                context = self._retriever.search(query, meeting_id)

        # Step 3: Compute confidence from retrieval quality
        confidence, confidence_score = self._compute_confidence(context, intent)

        # Step 3b: Guard against empty retrieval — do NOT send empty context to LLM
        has_chunks = len(context.chunks) > 0
        has_structured = (
            context.structured_context is not None
            and context.structured_context != "No structured data available."
        )

        if not has_chunks and not has_structured:
            latency = (time.time() - start_time) * 1000
            answer = self._build_no_results_answer(intent, speaker, meeting_id)
            logger.info(
                "query_no_results",
                intent=intent.value,
                speaker=speaker,
                meeting_id=meeting_id,
                latency_ms=round(latency),
            )
            return QueryResponse(
                answer=answer,
                sources=[],
                intent=intent,
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.0,
                tokens_used=0,
                latency_ms=round(latency, 1),
            )

        # Step 4: Build prompt with retrieved context
        formatted_context = context.format_context()
        user_prompt = QUERY_USER_PROMPT.format(
            context=formatted_context, question=query
        )

        # Step 5: Generate grounded answer
        try:
            response = await self._llm.generate(
                system_prompt=QUERY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except RateLimitError:
            # LLM rate-limited — return retrieved context directly instead of crashing
            latency = (time.time() - start_time) * 1000
            logger.warning(
                "query_rate_limited_fallback",
                intent=intent.value,
                chunks_retrieved=len(context.chunks),
            )
            fallback_answer = self._build_rate_limit_fallback(context)
            return QueryResponse(
                answer=fallback_answer,
                sources=context.to_sources(),
                intent=intent,
                confidence=confidence,
                confidence_score=confidence_score,
                tokens_used=0,
                latency_ms=round(latency, 1),
            )

        # Step 6: Build response with sources and metrics
        latency = (time.time() - start_time) * 1000

        logger.info(
            "query_processed",
            intent=intent.value,
            chunks_retrieved=len(context.chunks),
            latency_ms=round(latency),
            tokens_used=response.tokens_used,
        )

        return QueryResponse(
            answer=response.content,
            sources=context.to_sources(),
            intent=intent,
            confidence=confidence,
            confidence_score=confidence_score,
            tokens_used=response.tokens_used,
            latency_ms=round(latency, 1),
        )

    @staticmethod
    def _speaker_exists(speaker: str, known_speakers: set[str]) -> bool:
        """Check if an extracted speaker name matches any known speaker.

        Uses case-insensitive matching against individual name tokens so that
        "Sarah" matches "Sarah Johnson" and "tom" matches "Tom".
        """
        speaker_lower = speaker.lower()
        for known in known_speakers:
            known_lower = known.lower()
            # Exact match (case-insensitive)
            if speaker_lower == known_lower:
                return True
            # Token match — "Sarah" matches "Sarah Johnson"
            if speaker_lower in known_lower.split():
                return True
        return False

    @staticmethod
    def _build_speaker_not_found_answer(
        speaker: str, known_speakers: set[str], meeting_id: str | None
    ) -> str:
        """Build a definitive 'speaker not found' answer with the actual speaker list."""
        meeting_ctx = f" in meeting **{meeting_id}**" if meeting_id else " in any ingested meetings"

        if known_speakers:
            speaker_list = ", ".join(sorted(known_speakers))
            return (
                f"**{speaker}** does not appear as a speaker{meeting_ctx}. "
                f"The speakers in this meeting are: {speaker_list}."
            )

        return (
            f"**{speaker}** does not appear as a speaker{meeting_ctx}. "
            f"No speaker information is available."
        )

    @staticmethod
    def _build_no_results_answer(
        intent: QueryIntent, speaker: str | None, meeting_id: str | None
    ) -> str:
        """Build an honest 'no results' answer instead of sending empty context to LLM."""
        meeting_ctx = f" in meeting '{meeting_id}'" if meeting_id else " in any ingested meetings"

        if intent == QueryIntent.SPEAKER_SPECIFIC and speaker:
            return (
                f"No information found for speaker '{speaker}'{meeting_ctx}. "
                f"This person does not appear in the transcript. "
                f"Please check the speaker name and try again."
            )

        if intent == QueryIntent.STRUCTURED:
            return (
                f"No structured data (action items, decisions, summaries) found{meeting_ctx}."
            )

        return (
            f"I don't have any relevant information from the meeting transcripts to answer this question. "
            f"No matching content was found{meeting_ctx}."
        )

    @staticmethod
    def _compute_confidence(
        context: RetrievalResult, intent: QueryIntent
    ) -> tuple[ConfidenceLevel, float]:
        """Compute answer confidence from retrieval quality and query type.

        Returns:
            Tuple of (confidence level, numeric score 0.0-1.0).
        """
        has_structured = (
            context.structured_context is not None
            and context.structured_context != "No structured data available."
        )

        # Structured queries answered from SQLite are inherently reliable
        if intent == QueryIntent.STRUCTURED and has_structured:
            return ConfidenceLevel.HIGH, 1.0

        # For vector-search-based answers, use the top retrieval score
        if context.chunks:
            top_score = context.chunks[0].score

            # If retriever already flagged low confidence, cap at medium
            if context.low_confidence:
                if top_score >= 0.4:
                    return ConfidenceLevel.MEDIUM, round(top_score, 3)
                return ConfidenceLevel.LOW, round(top_score, 3)

            if top_score >= 0.7:
                return ConfidenceLevel.HIGH, round(top_score, 3)
            if top_score >= 0.4:
                return ConfidenceLevel.MEDIUM, round(top_score, 3)
            return ConfidenceLevel.LOW, round(top_score, 3)

        # Structured context exists but no chunks (mixed retrieval)
        if has_structured:
            return ConfidenceLevel.HIGH, 1.0

        return ConfidenceLevel.LOW, 0.0

    @staticmethod
    def _build_rate_limit_fallback(context: "RetrievalResult") -> str:
        """Build a fallback answer from raw context when LLM is rate-limited."""
        from src.models.schemas import RetrievalResult  # noqa: F811

        parts = ["**Note:** The AI model is temporarily rate-limited. "
                 "Here are the relevant transcript excerpts I found:\n"]

        if context.structured_context and context.structured_context != "No structured data available.":
            parts.append(context.structured_context)
            parts.append("")

        for chunk in context.chunks:
            speaker = chunk.metadata.speaker
            ts = chunk.metadata.timestamp_start or "N/A"
            parts.append(f"**{speaker}** (at {ts}):\n> {chunk.text}\n")

        return "\n".join(parts)

    def _match_known_speaker(self, query: str, meeting_id: str | None) -> str | None:
        """Try to match a speaker name from the query against known speakers.

        Handles cases where the regex-based extract_speaker fails but a known
        speaker name appears in the query text (e.g., "Was Tom in the meeting?").
        """
        known = self._retriever.get_known_speakers(meeting_id)
        query_lower = query.lower()
        for speaker in known:
            if speaker.lower() in query_lower:
                logger.info(
                    "speaker_matched_from_known_list",
                    speaker=speaker,
                    query_preview=query[:60],
                )
                return speaker
        return None

    async def _classify_intent(self, query: str) -> QueryIntent:
        """Classify the user query into an intent category.

        Tries LLM classification first. If LLM is unavailable (rate-limited),
        falls back to heuristic regex-based classification.
        """
        prompt = CLASSIFICATION_USER_PROMPT.format(question=query)

        try:
            response = await self._llm.generate(
                system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
            raw = response.content.strip().upper()

            # Extract the intent keyword from the response
            for key, intent in _INTENT_MAP.items():
                if key in raw:
                    logger.info("intent_classified", query_preview=query[:60], intent=intent.value)
                    return intent

            logger.warning("intent_classification_unclear", raw_response=raw)
            return QueryIntent.SEMANTIC

        except Exception:
            logger.exception("intent_classification_failed")
            # Fall back to heuristic classification
            return self._heuristic_classify(query)

    @staticmethod
    def _heuristic_classify(query: str) -> QueryIntent:
        """Regex/keyword fallback for intent classification when LLM is unavailable."""
        q = query.lower()

        # Speaker patterns
        if Retriever.extract_speaker(query):
            logger.info("heuristic_intent", intent="speaker", query_preview=query[:60])
            return QueryIntent.SPEAKER_SPECIFIC

        # Structured patterns
        structured_kw = ("action item", "todo", "task", "decision", "decided",
                         "summary", "summarize", "recap", "overview", "deadline",
                         "who attended", "who was present", "speaker", "participant",
                         "attendee")
        if any(kw in q for kw in structured_kw):
            logger.info("heuristic_intent", intent="structured", query_preview=query[:60])
            return QueryIntent.STRUCTURED

        # Cross-meeting patterns
        cross_kw = ("across meetings", "all meetings", "compare meetings",
                    "between meetings", "over time", "changed across")
        if any(kw in q for kw in cross_kw):
            logger.info("heuristic_intent", intent="cross_meeting", query_preview=query[:60])
            return QueryIntent.CROSS_MEETING

        logger.info("heuristic_intent", intent="semantic", query_preview=query[:60])
        return QueryIntent.SEMANTIC
