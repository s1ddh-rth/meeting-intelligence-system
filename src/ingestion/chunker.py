"""Speaker-aware chunker that preserves conversation structure."""

from __future__ import annotations

import re
from uuid import uuid4

import structlog

from src.models.schemas import Chunk, ChunkMetadata, Utterance

logger = structlog.get_logger()


def _estimate_tokens(text: str) -> int:
    """Approximate token count using word count * 1.3."""
    return int(len(text.split()) * 1.3)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at period, question mark, or exclamation boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


class SpeakerAwareChunker:
    """Chunks transcript utterances with speaker-aware boundaries.

    Chunking rules:
    1. Primary split on speaker change.
    2. Merge short consecutive turns from the same speaker (< min_chunk_tokens).
    3. Split long monologues (> max_chunk_tokens) at sentence boundaries.
    4. Attach rich metadata to every chunk.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 300,
        min_chunk_tokens: int = 30,
    ) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens

    def chunk(
        self, utterances: list[Utterance], meeting_id: str
    ) -> list[Chunk]:
        """Convert utterances into speaker-aware chunks.

        Args:
            utterances: Parsed transcript utterances.
            meeting_id: Identifier for this meeting.

        Returns:
            List of Chunk objects ready for embedding.
        """
        if not utterances:
            return []

        # Step 1: Group consecutive utterances by speaker
        groups = self._group_by_speaker(utterances)

        # Step 2: Merge short groups, split long ones, produce final chunks
        chunks: list[Chunk] = []
        chunk_index = 0

        for group in groups:
            speaker = group[0].speaker
            combined_text = " ".join(u.text for u in group)
            ts_start = group[0].timestamp
            ts_end = group[-1].timestamp
            token_count = _estimate_tokens(combined_text)

            if token_count <= self.max_chunk_tokens:
                # Fits in one chunk
                chunks.append(
                    self._make_chunk(
                        text=combined_text,
                        meeting_id=meeting_id,
                        speaker=speaker,
                        ts_start=ts_start,
                        ts_end=ts_end,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
            else:
                # Split long monologue at sentence boundaries
                sub_chunks = self._split_long_text(
                    combined_text, meeting_id, speaker, ts_start, ts_end, chunk_index
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

        logger.info(
            "chunking_complete",
            meeting_id=meeting_id,
            utterances=len(utterances),
            chunks=len(chunks),
        )
        return chunks

    def _group_by_speaker(
        self, utterances: list[Utterance]
    ) -> list[list[Utterance]]:
        """Group consecutive utterances by the same speaker.

        Merges short turns from the same speaker that are below min_chunk_tokens.
        """
        groups: list[list[Utterance]] = []
        current_group: list[Utterance] = [utterances[0]]

        for utt in utterances[1:]:
            prev_speaker = current_group[-1].speaker
            current_text = " ".join(u.text for u in current_group)
            current_tokens = _estimate_tokens(current_text)

            if utt.speaker == prev_speaker:
                # Same speaker — always merge
                current_group.append(utt)
            elif current_tokens < self.min_chunk_tokens and utt.speaker == current_group[0].speaker:
                # Very short chunk from same speaker after a brief interruption — merge
                current_group.append(utt)
            else:
                # Speaker changed — finalize group and start new one
                groups.append(current_group)
                current_group = [utt]

        groups.append(current_group)
        return groups

    def _split_long_text(
        self,
        text: str,
        meeting_id: str,
        speaker: str,
        ts_start: str | None,
        ts_end: str | None,
        start_index: int,
    ) -> list[Chunk]:
        """Split a long text block at sentence boundaries."""
        sentences = _split_sentences(text)
        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_tokens = 0
        idx = start_index

        for sentence in sentences:
            s_tokens = _estimate_tokens(sentence)

            if current_tokens + s_tokens > self.max_chunk_tokens and current_sentences:
                # Emit current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    self._make_chunk(
                        text=chunk_text,
                        meeting_id=meeting_id,
                        speaker=speaker,
                        ts_start=ts_start,
                        ts_end=ts_end,
                        chunk_index=idx,
                    )
                )
                idx += 1
                # Context overlap: carry last sentence forward
                current_sentences = [current_sentences[-1], sentence]
                current_tokens = _estimate_tokens(" ".join(current_sentences))
            else:
                current_sentences.append(sentence)
                current_tokens += s_tokens

        # Emit remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                self._make_chunk(
                    text=chunk_text,
                    meeting_id=meeting_id,
                    speaker=speaker,
                    ts_start=ts_start,
                    ts_end=ts_end,
                    chunk_index=idx,
                )
            )

        return chunks

    @staticmethod
    def _make_chunk(
        text: str,
        meeting_id: str,
        speaker: str,
        ts_start: str | None,
        ts_end: str | None,
        chunk_index: int,
    ) -> Chunk:
        """Create a Chunk with metadata."""
        return Chunk(
            id=str(uuid4()),
            text=text,
            metadata=ChunkMetadata(
                meeting_id=meeting_id,
                speaker=speaker,
                timestamp_start=ts_start,
                timestamp_end=ts_end,
                chunk_index=chunk_index,
                num_tokens=_estimate_tokens(text),
            ),
        )
