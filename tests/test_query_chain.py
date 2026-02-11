"""Tests for the query chain — uses mock LLM and retriever."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.chains.query_chain import QueryChain
from src.models.schemas import (
    ChunkMetadata,
    ConfidenceLevel,
    LLMResponse,
    QueryIntent,
    RetrievalResult,
    SearchResult,
)


@pytest.fixture
def mock_retriever() -> MagicMock:
    retriever = MagicMock()
    mock_result = RetrievalResult(
        chunks=[
            SearchResult(
                chunk_id="chunk-1",
                text="We should postpone the launch.",
                score=0.85,
                metadata=ChunkMetadata(
                    meeting_id="standup_2025_01_15",
                    speaker="Sarah",
                    timestamp_start="00:01:15",
                    timestamp_end=None,
                    chunk_index=0,
                    num_tokens=8,
                ),
            )
        ]
    )
    retriever.search.return_value = mock_result
    retriever.get_structured.return_value = mock_result
    retriever.extract_speaker = MagicMock(return_value="Sarah")
    retriever.get_known_speakers.return_value = {"Sarah", "Tom"}
    return retriever


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    return llm


@pytest.fixture
def query_chain(mock_retriever: MagicMock, mock_llm: AsyncMock) -> QueryChain:
    return QueryChain(retriever=mock_retriever, llm=mock_llm)


class TestIntentClassification:
    """Test that queries are classified to the correct intent."""

    @pytest.mark.asyncio
    async def test_structured_intent(
        self, query_chain: QueryChain, mock_llm: AsyncMock
    ) -> None:
        mock_llm.generate.return_value = LLMResponse(
            content="STRUCTURED", model="test", tokens_used=10
        )
        result = await query_chain.run("What are the action items?")
        assert result.intent == QueryIntent.STRUCTURED

    @pytest.mark.asyncio
    async def test_speaker_intent(
        self, query_chain: QueryChain, mock_llm: AsyncMock
    ) -> None:
        mock_llm.generate.return_value = LLMResponse(
            content="SPEAKER", model="test", tokens_used=10
        )
        result = await query_chain.run("What did Sarah say?")
        assert result.intent == QueryIntent.SPEAKER_SPECIFIC

    @pytest.mark.asyncio
    async def test_semantic_intent(
        self, query_chain: QueryChain, mock_llm: AsyncMock
    ) -> None:
        mock_llm.generate.return_value = LLMResponse(
            content="SEMANTIC", model="test", tokens_used=10
        )
        result = await query_chain.run("Tell me about the API discussion")
        assert result.intent == QueryIntent.SEMANTIC

    @pytest.mark.asyncio
    async def test_cross_meeting_intent(
        self, query_chain: QueryChain, mock_llm: AsyncMock
    ) -> None:
        mock_llm.generate.return_value = LLMResponse(
            content="CROSS_MEETING", model="test", tokens_used=10
        )
        result = await query_chain.run("How has the timeline changed across meetings?")
        assert result.intent == QueryIntent.CROSS_MEETING

    @pytest.mark.asyncio
    async def test_fallback_to_semantic(
        self, query_chain: QueryChain, mock_llm: AsyncMock
    ) -> None:
        mock_llm.generate.return_value = LLMResponse(
            content="UNKNOWN_GIBBERISH", model="test", tokens_used=10
        )
        result = await query_chain.run("Some random question")
        assert result.intent == QueryIntent.SEMANTIC


class TestQueryRouting:
    """Test that intents route to the correct retrieval method."""

    @pytest.mark.asyncio
    async def test_structured_uses_get_structured(
        self, query_chain: QueryChain, mock_llm: AsyncMock, mock_retriever: MagicMock
    ) -> None:
        mock_llm.generate.return_value = LLMResponse(
            content="STRUCTURED", model="test", tokens_used=10
        )
        await query_chain.run("What are the action items?")
        mock_retriever.get_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_speaker_uses_search_with_filter(
        self, query_chain: QueryChain, mock_llm: AsyncMock, mock_retriever: MagicMock
    ) -> None:
        mock_llm.generate.return_value = LLMResponse(
            content="SPEAKER", model="test", tokens_used=10
        )
        await query_chain.run("What did Sarah say?")
        mock_retriever.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_response_has_sources_and_metrics(
        self, query_chain: QueryChain, mock_llm: AsyncMock
    ) -> None:
        mock_llm.generate.return_value = LLMResponse(
            content="SEMANTIC", model="test", tokens_used=10
        )
        result = await query_chain.run("Tell me about the launch")
        assert result.answer is not None
        assert result.latency_ms >= 0
        assert len(result.sources) > 0


class TestNoResultsGuard:
    """Test that empty retrieval returns honest answers without calling LLM."""

    @pytest.mark.asyncio
    async def test_speaker_not_found_returns_honest_answer(
        self, mock_llm: AsyncMock
    ) -> None:
        """When a speaker doesn't exist, return a direct answer without LLM."""
        retriever = MagicMock()
        retriever.search.return_value = RetrievalResult(chunks=[])
        retriever.extract_speaker = MagicMock(return_value="Sid")
        retriever.get_known_speakers.return_value = {"Sarah", "Tom"}

        chain = QueryChain(retriever=retriever, llm=mock_llm)

        # First call = intent classification, must return SPEAKER
        mock_llm.generate.return_value = LLMResponse(
            content="SPEAKER", model="test", tokens_used=10
        )

        result = await chain.run("What did Sid say?", meeting_id="standup_2025_01_15")

        assert result.intent == QueryIntent.SPEAKER_SPECIFIC
        assert "Sid" in result.answer
        assert "does not appear" in result.answer
        assert "Sarah" in result.answer  # Shows actual speakers
        assert "Tom" in result.answer
        assert len(result.sources) == 0
        assert result.confidence == ConfidenceLevel.HIGH  # Definitive answer
        # LLM should only be called once (for classification), NOT for answer generation
        assert mock_llm.generate.call_count == 1
        # Vector search should NOT be called — short-circuited before it
        retriever.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_results_semantic_returns_honest_answer(
        self, mock_llm: AsyncMock
    ) -> None:
        """When semantic search finds nothing, return direct answer."""
        retriever = MagicMock()
        retriever.search.return_value = RetrievalResult(chunks=[])

        chain = QueryChain(retriever=retriever, llm=mock_llm)

        mock_llm.generate.return_value = LLMResponse(
            content="SEMANTIC", model="test", tokens_used=10
        )

        result = await chain.run("Tell me about quantum physics")

        assert "don't have any relevant information" in result.answer
        assert len(result.sources) == 0
        assert mock_llm.generate.call_count == 1  # Only classification, no answer gen


class TestSpeakerExistenceCheck:
    """Test the fast-path speaker existence check against known speakers."""

    def test_exact_match(self) -> None:
        assert QueryChain._speaker_exists("Sarah", {"Sarah", "Tom"}) is True

    def test_case_insensitive(self) -> None:
        assert QueryChain._speaker_exists("sarah", {"Sarah", "Tom"}) is True
        assert QueryChain._speaker_exists("SARAH", {"Sarah", "Tom"}) is True

    def test_partial_name_matches_full_name(self) -> None:
        """'Sarah' should match 'Sarah Johnson'."""
        assert QueryChain._speaker_exists("Sarah", {"Sarah Johnson", "Tom"}) is True

    def test_last_name_matches_full_name(self) -> None:
        """'Johnson' should match 'Sarah Johnson'."""
        assert QueryChain._speaker_exists("Johnson", {"Sarah Johnson", "Tom"}) is True

    def test_nonexistent_speaker(self) -> None:
        assert QueryChain._speaker_exists("Sid", {"Sarah", "Tom"}) is False

    def test_empty_known_speakers(self) -> None:
        assert QueryChain._speaker_exists("Sarah", set()) is False

    def test_partial_string_not_a_token_does_not_match(self) -> None:
        """'Sara' should NOT match 'Sarah' — must be a full token match."""
        assert QueryChain._speaker_exists("Sara", {"Sarah", "Tom"}) is False

    @pytest.mark.asyncio
    async def test_existing_speaker_proceeds_to_search(
        self, mock_llm: AsyncMock
    ) -> None:
        """When speaker exists, the normal vector search path runs."""
        retriever = MagicMock()
        mock_result = RetrievalResult(
            chunks=[
                SearchResult(
                    chunk_id="chunk-1",
                    text="I think we should postpone.",
                    score=0.85,
                    metadata=ChunkMetadata(
                        meeting_id="standup_2025_01_15",
                        speaker="Sarah",
                        timestamp_start="00:01:15",
                        timestamp_end=None,
                        chunk_index=0,
                        num_tokens=8,
                    ),
                )
            ]
        )
        retriever.search.return_value = mock_result
        retriever.extract_speaker = MagicMock(return_value="Sarah")
        retriever.get_known_speakers.return_value = {"Sarah", "Tom"}

        chain = QueryChain(retriever=retriever, llm=mock_llm)
        mock_llm.generate.return_value = LLMResponse(
            content="SPEAKER", model="test", tokens_used=10
        )

        result = await chain.run("What did Sarah say?")

        # Should proceed to vector search, not short-circuit
        retriever.search.assert_called_once()
        assert len(result.sources) > 0

    @pytest.mark.asyncio
    async def test_nonexistent_speaker_skips_search_and_llm(
        self, mock_llm: AsyncMock
    ) -> None:
        """Non-existent speaker short-circuits: no vector search, no answer LLM call."""
        retriever = MagicMock()
        retriever.extract_speaker = MagicMock(return_value="Sid")
        retriever.get_known_speakers.return_value = {"Sarah", "Tom"}

        chain = QueryChain(retriever=retriever, llm=mock_llm)
        mock_llm.generate.return_value = LLMResponse(
            content="SPEAKER", model="test", tokens_used=10
        )

        result = await chain.run("Was Sid in this meeting?", meeting_id="design_review")

        assert "Sid" in result.answer
        assert "does not appear" in result.answer
        assert result.confidence == ConfidenceLevel.HIGH
        retriever.search.assert_not_called()
        # Only 1 LLM call (classification), not 2 (classification + answer)
        assert mock_llm.generate.call_count == 1
