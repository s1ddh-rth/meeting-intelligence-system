"""Shared test fixtures."""

from __future__ import annotations

import pytest

from src.ingestion.parser import TranscriptParser
from src.ingestion.chunker import SpeakerAwareChunker
from src.models.schemas import Utterance


@pytest.fixture
def parser() -> TranscriptParser:
    return TranscriptParser()


@pytest.fixture
def chunker() -> SpeakerAwareChunker:
    return SpeakerAwareChunker(max_chunk_tokens=300, min_chunk_tokens=30)


@pytest.fixture
def timestamped_transcript() -> str:
    return (
        "[00:01:15] Sarah: I think we should postpone the launch.\n"
        "[00:01:32] Tom: I agree. The API isn't ready.\n"
        "[00:01:45] Sarah: Let's push it to next week.\n"
        "[00:02:00] Tom: That works. I'll update the timeline.\n"
    )


@pytest.fixture
def simple_transcript() -> str:
    return (
        "Sarah: I think we should postpone the launch.\n"
        "Tom: I agree. The API isn't ready.\n"
        "Sarah: Let's push it to next week.\n"
        "Tom: That works. I'll update the timeline.\n"
    )


@pytest.fixture
def sample_utterances() -> list[Utterance]:
    return [
        Utterance(speaker="Sarah", timestamp="00:01:15", text="I think we should postpone the launch."),
        Utterance(speaker="Tom", timestamp="00:01:32", text="I agree. The API isn't ready."),
        Utterance(speaker="Sarah", timestamp="00:01:45", text="Let's push it to next week."),
        Utterance(speaker="Tom", timestamp="00:02:00", text="That works. I'll update the timeline."),
    ]
