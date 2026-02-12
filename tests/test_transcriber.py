"""Tests for the AudioTranscriber module."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.voice.transcriber import AudioTranscriber


# --- Helpers / fakes ---


@dataclass
class FakeSegment:
    """Mimics a faster-whisper segment namedtuple."""

    start: float
    end: float
    text: str


@dataclass
class FakeTranscriptionInfo:
    """Mimics the info object returned by faster-whisper."""

    language: str = "en"
    language_probability: float = 0.98


@dataclass
class FakeTurn:
    """Mimics a pyannote timeline turn."""

    start: float
    end: float


class FakeDiarization:
    """Mimics pyannote diarization output with itertracks()."""

    def __init__(self, tracks: list[tuple[FakeTurn, str, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False) -> list[tuple[FakeTurn, str, str]]:
        return self._tracks


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _mock_torchaudio() -> None:
    """Mock torchaudio module — only available inside Docker (CPU wheel)."""
    mock_waveform = MagicMock()
    mock_waveform.shape = (1, 16000)  # mono, 1 second at 16kHz
    mock_waveform.mean.return_value = mock_waveform

    mock_module = MagicMock()
    mock_module.load.return_value = (mock_waveform, 16000)
    mock_module.functional.resample.return_value = mock_waveform

    with patch.dict(sys.modules, {"torchaudio": mock_module}):
        yield


@pytest.fixture
def transcriber() -> AudioTranscriber:
    return AudioTranscriber(whisper_model="base", hf_token="test-token")


@pytest.fixture
def fake_segments() -> list[FakeSegment]:
    return [
        FakeSegment(start=0.0, end=3.5, text=" Hello everyone, let's get started."),
        FakeSegment(start=4.0, end=7.2, text=" I have the quarterly numbers ready."),
        FakeSegment(start=8.0, end=12.0, text=" Great, can you share your screen?"),
    ]


@pytest.fixture
def fake_diarization() -> FakeDiarization:
    return FakeDiarization([
        (FakeTurn(start=0.0, end=4.0), "", "SPEAKER_00"),
        (FakeTurn(start=4.0, end=8.0), "", "SPEAKER_01"),
        (FakeTurn(start=8.0, end=12.0), "", "SPEAKER_00"),
    ])


# --- Unit tests ---


class TestFormatTimestamp:
    """Tests for _format_timestamp static method."""

    def test_zero(self) -> None:
        assert AudioTranscriber._format_timestamp(0.0) == "00:00:00"

    def test_seconds_only(self) -> None:
        assert AudioTranscriber._format_timestamp(45.7) == "00:00:45"

    def test_minutes_and_seconds(self) -> None:
        assert AudioTranscriber._format_timestamp(125.0) == "00:02:05"

    def test_hours(self) -> None:
        assert AudioTranscriber._format_timestamp(3661.0) == "01:01:01"

    def test_large_value(self) -> None:
        assert AudioTranscriber._format_timestamp(7200.0) == "02:00:00"


class TestFindSpeaker:
    """Tests for _find_speaker static method."""

    def test_exact_overlap(self, fake_diarization: FakeDiarization) -> None:
        speaker = AudioTranscriber._find_speaker(fake_diarization, 0.0, 3.5)
        assert speaker == "SPEAKER_00"

    def test_second_speaker(self, fake_diarization: FakeDiarization) -> None:
        speaker = AudioTranscriber._find_speaker(fake_diarization, 4.0, 7.2)
        assert speaker == "SPEAKER_01"

    def test_no_overlap_returns_unknown(self) -> None:
        empty_diarization = FakeDiarization([])
        speaker = AudioTranscriber._find_speaker(empty_diarization, 0.0, 5.0)
        assert speaker == "Unknown"

    def test_best_overlap_wins(self) -> None:
        """When a segment spans two speaker turns, the one with more overlap wins."""
        diarization = FakeDiarization([
            (FakeTurn(start=0.0, end=3.0), "", "SPEAKER_00"),
            (FakeTurn(start=3.0, end=10.0), "", "SPEAKER_01"),
        ])
        # Segment from 2.0 to 5.0: overlap with SPEAKER_00 = 1.0s, SPEAKER_01 = 2.0s
        speaker = AudioTranscriber._find_speaker(diarization, 2.0, 5.0)
        assert speaker == "SPEAKER_01"


class TestTranscribe:
    """Tests for the full transcribe pipeline (with mocked models)."""

    def test_transcribe_produces_format_a(
        self,
        transcriber: AudioTranscriber,
        fake_segments: list[FakeSegment],
        fake_diarization: FakeDiarization,
    ) -> None:
        """Transcription output matches Format A: [HH:MM:SS] Speaker N: text."""
        fake_info = FakeTranscriptionInfo()

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = (fake_segments, fake_info)
        transcriber._whisper_model = mock_whisper

        mock_pipeline = MagicMock(return_value=fake_diarization)
        transcriber._diarization_pipeline = mock_pipeline

        result = transcriber.transcribe("/fake/audio.wav")

        lines = result.strip().split("\n")
        assert len(lines) == 3

        # Check Format A pattern: [HH:MM:SS] Speaker N: text
        assert lines[0] == "[00:00:00] Speaker 1: Hello everyone, let's get started."
        assert lines[1] == "[00:00:04] Speaker 2: I have the quarterly numbers ready."
        assert lines[2] == "[00:00:08] Speaker 1: Great, can you share your screen?"

    def test_speaker_renaming(
        self,
        transcriber: AudioTranscriber,
        fake_segments: list[FakeSegment],
        fake_diarization: FakeDiarization,
    ) -> None:
        """SPEAKER_00 → Speaker 1, SPEAKER_01 → Speaker 2."""
        fake_info = FakeTranscriptionInfo()

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = (fake_segments, fake_info)
        transcriber._whisper_model = mock_whisper

        mock_pipeline = MagicMock(return_value=fake_diarization)
        transcriber._diarization_pipeline = mock_pipeline

        result = transcriber.transcribe("/fake/audio.wav")

        assert "Speaker 1:" in result
        assert "Speaker 2:" in result
        assert "SPEAKER_00" not in result
        assert "SPEAKER_01" not in result

    def test_empty_segments_produce_empty_output(
        self,
        transcriber: AudioTranscriber,
    ) -> None:
        """If whisper returns no segments, transcription is empty."""
        fake_info = FakeTranscriptionInfo()

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = ([], fake_info)
        transcriber._whisper_model = mock_whisper

        empty_diarization = FakeDiarization([])
        mock_pipeline = MagicMock(return_value=empty_diarization)
        transcriber._diarization_pipeline = mock_pipeline

        result = transcriber.transcribe("/fake/audio.wav")
        assert result == ""

    def test_blank_text_segments_skipped(
        self,
        transcriber: AudioTranscriber,
    ) -> None:
        """Segments with whitespace-only text are skipped."""
        segments = [
            FakeSegment(start=0.0, end=2.0, text="   "),
            FakeSegment(start=2.0, end=5.0, text=" Actual text."),
        ]
        fake_info = FakeTranscriptionInfo()

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = (segments, fake_info)
        transcriber._whisper_model = mock_whisper

        diarization = FakeDiarization([
            (FakeTurn(start=0.0, end=5.0), "", "SPEAKER_00"),
        ])
        mock_pipeline = MagicMock(return_value=diarization)
        transcriber._diarization_pipeline = mock_pipeline

        result = transcriber.transcribe("/fake/audio.wav")
        lines = result.strip().split("\n")
        assert len(lines) == 1
        assert "Actual text." in lines[0]


class TestLazyLoading:
    """Tests for lazy model loading behaviour."""

    def test_models_not_loaded_on_init(self, transcriber: AudioTranscriber) -> None:
        """Models should be None until transcribe() is called."""
        assert transcriber._whisper_model is None
        assert transcriber._diarization_pipeline is None

    @patch("src.voice.transcriber.AudioTranscriber._load_models")
    def test_load_models_called_on_transcribe(
        self,
        mock_load: MagicMock,
        transcriber: AudioTranscriber,
    ) -> None:
        """_load_models is called when transcribe is invoked."""
        # Set up pre-loaded mocks so transcribe doesn't fail after _load_models
        fake_info = FakeTranscriptionInfo()
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = ([], fake_info)
        transcriber._whisper_model = mock_whisper

        empty_diarization = FakeDiarization([])
        mock_pipeline = MagicMock(return_value=empty_diarization)
        transcriber._diarization_pipeline = mock_pipeline

        transcriber.transcribe("/fake/audio.wav")
        mock_load.assert_called_once()

    def test_missing_hf_token_raises_on_load(self) -> None:
        """ValueError raised if HF_TOKEN is empty when loading pyannote."""
        t = AudioTranscriber(whisper_model="base", hf_token="")

        # Pre-load whisper so we only test pyannote path
        t._whisper_model = MagicMock()

        with pytest.raises(ValueError, match="HF_TOKEN"):
            t._load_models()
