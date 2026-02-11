"""Tests for the transcript parser."""

from __future__ import annotations

from src.ingestion.parser import TranscriptParser


class TestTimestampedFormat:
    """Test Format A: timestamped transcripts."""

    def test_parse_basic_timestamped(self, parser: TranscriptParser) -> None:
        content = (
            "[00:01:15] Sarah: I think we should postpone the launch.\n"
            "[00:01:32] Tom: I agree. The API isn't ready.\n"
        )
        result = parser.parse(content)
        assert len(result) == 2
        assert result[0].speaker == "Sarah"
        assert result[0].timestamp == "00:01:15"
        assert result[0].text == "I think we should postpone the launch."
        assert result[1].speaker == "Tom"
        assert result[1].timestamp == "00:01:32"

    def test_parse_short_timestamp(self, parser: TranscriptParser) -> None:
        content = "[1:05] Sarah: Quick point.\n"
        result = parser.parse(content)
        assert len(result) == 1
        assert result[0].timestamp == "1:05"

    def test_parse_full_transcript(self, parser: TranscriptParser, timestamped_transcript: str) -> None:
        result = parser.parse(timestamped_transcript)
        assert len(result) == 4
        speakers = {u.speaker for u in result}
        assert speakers == {"Sarah", "Tom"}


class TestSimpleFormat:
    """Test Format B: simple speaker labels."""

    def test_parse_basic_simple(self, parser: TranscriptParser) -> None:
        content = (
            "Sarah: I think we should postpone the launch.\n"
            "Tom: I agree.\n"
        )
        result = parser.parse(content)
        assert len(result) == 2
        assert result[0].speaker == "Sarah"
        assert result[0].timestamp is None
        assert result[0].text == "I think we should postpone the launch."

    def test_parse_full_simple_transcript(self, parser: TranscriptParser, simple_transcript: str) -> None:
        result = parser.parse(simple_transcript)
        assert len(result) == 4


class TestSpeakerNormalisation:
    """Test speaker name normalisation."""

    def test_whitespace_stripping(self, parser: TranscriptParser) -> None:
        content = "  sarah  : Hello everyone.\n"
        result = parser.parse(content)
        assert len(result) == 1
        assert result[0].speaker == "Sarah"

    def test_title_case(self, parser: TranscriptParser) -> None:
        content = "tom: Hello.\nsarah: Hi.\n"
        result = parser.parse(content)
        assert result[0].speaker == "Tom"
        assert result[1].speaker == "Sarah"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self, parser: TranscriptParser) -> None:
        result = parser.parse("")
        assert result == []

    def test_only_empty_lines(self, parser: TranscriptParser) -> None:
        result = parser.parse("\n\n\n")
        assert result == []

    def test_multiline_utterance(self, parser: TranscriptParser) -> None:
        content = (
            "Sarah: This is the first line.\n"
            "And this is a continuation.\n"
            "Tom: OK, got it.\n"
        )
        result = parser.parse(content)
        assert len(result) == 2
        assert "continuation" in result[0].text
        assert result[1].speaker == "Tom"

    def test_empty_lines_between_speakers(self, parser: TranscriptParser) -> None:
        content = (
            "Sarah: Hello.\n"
            "\n"
            "\n"
            "Tom: Hi there.\n"
        )
        result = parser.parse(content)
        assert len(result) == 2
