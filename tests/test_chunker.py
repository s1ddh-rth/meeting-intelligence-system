"""Tests for the speaker-aware chunker."""

from __future__ import annotations

from src.ingestion.chunker import SpeakerAwareChunker, _estimate_tokens
from src.models.schemas import Utterance


class TestSpeakerChangeSplitting:
    """Test that chunks split on speaker boundaries."""

    def test_different_speakers_produce_separate_chunks(
        self, chunker: SpeakerAwareChunker, sample_utterances: list[Utterance]
    ) -> None:
        chunks = chunker.chunk(sample_utterances, "test_meeting")
        # Sarah, Tom, Sarah, Tom should produce at least 2 chunks (may merge short same-speaker turns)
        assert len(chunks) >= 2
        # Verify each chunk has correct speaker metadata
        for chunk in chunks:
            assert chunk.metadata.speaker in ("Sarah", "Tom")

    def test_single_speaker_produces_one_chunk(self, chunker: SpeakerAwareChunker) -> None:
        utterances = [
            Utterance(speaker="Sarah", text="First point."),
            Utterance(speaker="Sarah", text="Second point."),
            Utterance(speaker="Sarah", text="Third point."),
        ]
        chunks = chunker.chunk(utterances, "test_meeting")
        assert len(chunks) == 1
        assert chunks[0].metadata.speaker == "Sarah"


class TestShortTurnMerging:
    """Test merging of short consecutive same-speaker turns."""

    def test_short_turns_from_same_speaker_merge(self) -> None:
        chunker = SpeakerAwareChunker(min_chunk_tokens=30)
        utterances = [
            Utterance(speaker="Tom", text="Yes."),
            Utterance(speaker="Tom", text="I agree."),
        ]
        chunks = chunker.chunk(utterances, "test_meeting")
        assert len(chunks) == 1
        assert "Yes." in chunks[0].text
        assert "I agree." in chunks[0].text


class TestLongMonologueSplitting:
    """Test splitting of long monologues at sentence boundaries."""

    def test_long_monologue_is_split(self) -> None:
        chunker = SpeakerAwareChunker(max_chunk_tokens=20)
        # Create a long monologue that exceeds max_chunk_tokens
        long_text = ". ".join([f"This is sentence number {i}" for i in range(20)]) + "."
        utterances = [Utterance(speaker="Sarah", text=long_text)]
        chunks = chunker.chunk(utterances, "test_meeting")
        assert len(chunks) > 1
        # All chunks should be from Sarah
        for chunk in chunks:
            assert chunk.metadata.speaker == "Sarah"


class TestMetadata:
    """Test that metadata is correctly attached to chunks."""

    def test_meeting_id_set(
        self, chunker: SpeakerAwareChunker, sample_utterances: list[Utterance]
    ) -> None:
        chunks = chunker.chunk(sample_utterances, "standup_2025_01_15")
        for chunk in chunks:
            assert chunk.metadata.meeting_id == "standup_2025_01_15"

    def test_chunk_index_sequential(
        self, chunker: SpeakerAwareChunker, sample_utterances: list[Utterance]
    ) -> None:
        chunks = chunker.chunk(sample_utterances, "test_meeting")
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == sorted(indices)

    def test_token_count_positive(
        self, chunker: SpeakerAwareChunker, sample_utterances: list[Utterance]
    ) -> None:
        chunks = chunker.chunk(sample_utterances, "test_meeting")
        for chunk in chunks:
            assert chunk.metadata.num_tokens > 0

    def test_timestamps_preserved(self, chunker: SpeakerAwareChunker) -> None:
        utterances = [
            Utterance(speaker="Sarah", timestamp="00:01:15", text="Hello."),
            Utterance(speaker="Tom", timestamp="00:01:30", text="Hi."),
        ]
        chunks = chunker.chunk(utterances, "test_meeting")
        assert chunks[0].metadata.timestamp_start == "00:01:15"


class TestTokenEstimation:
    """Test the token estimation helper."""

    def test_empty_string(self) -> None:
        assert _estimate_tokens("") == 0

    def test_single_word(self) -> None:
        assert _estimate_tokens("hello") == 1  # 1 * 1.3 = 1.3 → int = 1

    def test_approximate_count(self) -> None:
        text = "this is a test sentence with seven words"
        tokens = _estimate_tokens(text)
        assert 7 <= tokens <= 15  # Reasonable range
