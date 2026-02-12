"""Transcript parser supporting timestamped and simple speaker-label formats."""

from __future__ import annotations

import re

import structlog

from src.models.schemas import Utterance

logger = structlog.get_logger()

# Pattern A: [HH:MM:SS] Speaker: text
_TIMESTAMPED_PATTERN = re.compile(
    r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s+([^:]+?):\s*(.+)$"
)

# Pattern B: Speaker: text (no timestamp)
_SIMPLE_PATTERN = re.compile(r"^([^:\[\]]{1,50}):\s*(.+)$")


class TranscriptParser:
    """Parses meeting transcripts into a list of Utterance objects.

    Supports two formats:
    - Format A (timestamped): [00:01:15] Sarah: I think we should...
    - Format B (simple labels): Sarah: I think we should...

    Auto-detects the format and handles multi-line utterances, empty lines,
    and continuation lines (lines without a speaker label).
    """

    def parse(self, content: str) -> list[Utterance]:
        """Parse raw transcript text into utterances.

        Args:
            content: Raw transcript file content.

        Returns:
            List of parsed Utterance objects.
        """
        lines = content.strip().splitlines()
        if not lines:
            logger.warning("empty_transcript")
            return []

        is_timestamped = self._detect_format(lines)
        logger.info(
            "transcript_format_detected",
            format="timestamped" if is_timestamped else "simple",
            total_lines=len(lines),
        )

        utterances: list[Utterance] = []
        current: Utterance | None = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            parsed = self._parse_line(stripped, is_timestamped)

            if parsed is not None:
                # New speaker turn — save previous and start new
                if current is not None:
                    utterances.append(current)
                current = parsed
            elif current is not None:
                # Continuation line — append to current speaker's text
                current = Utterance(
                    speaker=current.speaker,
                    timestamp=current.timestamp,
                    text=f"{current.text} {stripped}",
                )

        # Don't forget the last utterance
        if current is not None:
            utterances.append(current)

        logger.info(
            "transcript_parsed",
            utterances=len(utterances),
            speakers=list({u.speaker for u in utterances}),
        )
        return utterances

    def _detect_format(self, lines: list[str]) -> bool:
        """Check if the transcript uses timestamped format.

        Scans the first 10 non-empty lines for timestamp patterns.
        """
        non_empty = [line.strip() for line in lines if line.strip()][:10]
        timestamped_count = sum(
            1 for line in non_empty if _TIMESTAMPED_PATTERN.match(line)
        )
        return timestamped_count >= len(non_empty) * 0.5

    def _parse_line(
        self, line: str, is_timestamped: bool
    ) -> Utterance | None:
        """Attempt to parse a single line into an Utterance.

        Returns None if the line is a continuation (no speaker label).
        """
        if is_timestamped:
            match = _TIMESTAMPED_PATTERN.match(line)
            if match:
                timestamp, speaker, text = match.groups()
                return Utterance(
                    speaker=self._normalise_speaker(speaker),
                    timestamp=timestamp,
                    text=text.strip(),
                )
        else:
            match = _SIMPLE_PATTERN.match(line)
            if match:
                speaker, text = match.groups()
                # Avoid matching lines that look like metadata, not speaker turns
                if not speaker.strip().startswith(("[", "#", "//")):
                    return Utterance(
                        speaker=self._normalise_speaker(speaker),
                        timestamp=None,
                        text=text.strip(),
                    )

        return None

    @staticmethod
    def _normalise_speaker(name: str) -> str:
        """Normalise speaker names: strip whitespace and title-case."""
        return name.strip().title()
