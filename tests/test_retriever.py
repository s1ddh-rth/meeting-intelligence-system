"""Tests for the retriever — uses mock embedder and stores."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.models.schemas import (
    ActionItem,
    ChunkMetadata,
    Decision,
    SearchResult,
)
from src.retrieval.retriever import Retriever


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 384
    return embedder


@pytest.fixture
def mock_vector_store() -> MagicMock:
    store = MagicMock()
    store.search.return_value = [
        SearchResult(
            chunk_id="chunk-1",
            text="I think we should postpone the launch.",
            score=0.85,
            metadata=ChunkMetadata(
                meeting_id="standup_2025_01_15",
                speaker="Sarah",
                timestamp_start="00:01:15",
                timestamp_end="00:01:15",
                chunk_index=0,
                num_tokens=10,
            ),
        ),
        SearchResult(
            chunk_id="chunk-2",
            text="The API isn't ready.",
            score=0.72,
            metadata=ChunkMetadata(
                meeting_id="standup_2025_01_15",
                speaker="Tom",
                timestamp_start="00:01:32",
                timestamp_end="00:01:32",
                chunk_index=1,
                num_tokens=5,
            ),
        ),
    ]
    return store


@pytest.fixture
def mock_structured_store() -> MagicMock:
    store = MagicMock()
    store.get_action_items.return_value = [
        ActionItem(assignee="Tom", task="Fix rollback script", deadline=None),
        ActionItem(assignee="Maya", task="Accessibility audit", deadline="Thursday"),
    ]
    store.get_decisions.return_value = [
        Decision(decision="Postpone launch", context="API not ready", decided_by=["Sarah", "Tom"]),
    ]
    store.get_meeting_summary.return_value = "Team discussed launch timing and blockers."
    store.list_meetings.return_value = []
    return store


@pytest.fixture
def retriever(
    mock_embedder: MagicMock,
    mock_vector_store: MagicMock,
    mock_structured_store: MagicMock,
) -> Retriever:
    return Retriever(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        structured_store=mock_structured_store,
        top_k=5,
        similarity_threshold=0.3,
    )


class TestVectorSearch:
    """Test vector similarity search."""

    def test_search_returns_ranked_results(self, retriever: Retriever) -> None:
        result = retriever.search("launch timeline")
        assert len(result.chunks) == 2
        assert result.chunks[0].score >= result.chunks[1].score

    def test_search_with_speaker_filter(
        self, retriever: Retriever, mock_vector_store: MagicMock
    ) -> None:
        retriever.search("launch", speaker_filter="Sarah")
        mock_vector_store.search.assert_called_once()
        call_kwargs = mock_vector_store.search.call_args
        assert call_kwargs.kwargs.get("speaker") == "Sarah" or call_kwargs[1].get("speaker") == "Sarah"

    def test_search_with_meeting_filter(
        self, retriever: Retriever, mock_vector_store: MagicMock
    ) -> None:
        retriever.search("launch", meeting_id="standup_2025_01_15")
        call_kwargs = mock_vector_store.search.call_args
        assert (
            call_kwargs.kwargs.get("meeting_id") == "standup_2025_01_15"
            or call_kwargs[1].get("meeting_id") == "standup_2025_01_15"
        )


class TestStructuredQuery:
    """Test structured retrieval from SQLite."""

    def test_get_structured_returns_action_items(self, retriever: Retriever) -> None:
        result = retriever.get_structured("What are the action items?")
        assert result.structured_context is not None
        assert "ACTION ITEMS" in result.structured_context
        assert "Tom" in result.structured_context

    def test_get_structured_returns_decisions(self, retriever: Retriever) -> None:
        result = retriever.get_structured("What decisions were made?")
        assert result.structured_context is not None
        assert "DECISIONS" in result.structured_context

    def test_get_structured_skips_vector_search_when_data_available(self, retriever: Retriever) -> None:
        result = retriever.get_structured("What are the action items?")
        # When structured data is found, vector search should be skipped
        assert len(result.chunks) == 0
        assert "ACTION ITEMS" in result.structured_context


class TestSpeakerExtraction:
    """Test speaker name extraction from queries."""

    def test_what_did_pattern(self) -> None:
        assert Retriever.extract_speaker("What did Sarah say about the launch?") == "Sarah"

    def test_possessive_pattern(self) -> None:
        assert Retriever.extract_speaker("What were Tom's concerns?") == "Tom"

    def test_no_speaker(self) -> None:
        assert Retriever.extract_speaker("What was discussed about the API?") is None

    def test_according_to_pattern(self) -> None:
        assert Retriever.extract_speaker("According to Maya, when is the deadline?") == "Maya"
