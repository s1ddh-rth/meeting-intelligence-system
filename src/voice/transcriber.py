"""Audio transcription using Whisper + Pyannote for speaker diarization.

BONUS MODULE — Optional. Requires additional dependencies:
    pip install openai-whisper pyannote.audio torchaudio

This module converts audio files (.wav, .mp3) into timestamped, speaker-labeled
transcripts compatible with the existing parser pipeline (Format A).
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()


class AudioTranscriber:
    """Converts audio to speaker-labelled transcript using Whisper + Pyannote.

    Combines OpenAI Whisper (speech-to-text) with Pyannote (speaker diarization)
    by aligning Whisper segments with Pyannote speaker labels via temporal overlap.

    The output is a formatted transcript string matching Format A (timestamped),
    which feeds directly into the existing TranscriptParser.
    """

    def __init__(self, whisper_model: str = "base") -> None:
        self._whisper_model_name = whisper_model
        self._whisper_model = None
        self._diarization_pipeline = None

    def _load_models(self) -> None:
        """Lazy-load Whisper and Pyannote models."""
        if self._whisper_model is None:
            try:
                import whisper

                logger.info("loading_whisper_model", model=self._whisper_model_name)
                self._whisper_model = whisper.load_model(self._whisper_model_name)
            except ImportError:
                raise ImportError(
                    "openai-whisper is required for audio transcription. "
                    "Install with: pip install openai-whisper"
                )

        if self._diarization_pipeline is None:
            try:
                from pyannote.audio import Pipeline

                logger.info("loading_pyannote_pipeline")
                self._diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                )
            except ImportError:
                raise ImportError(
                    "pyannote.audio is required for speaker diarization. "
                    "Install with: pip install pyannote.audio"
                )

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to a speaker-labelled transcript.

        Args:
            audio_path: Path to .wav or .mp3 file.

        Returns:
            Formatted transcript string (Format A: [HH:MM:SS] Speaker: text).
        """
        self._load_models()

        logger.info("transcription_started", audio_path=audio_path)

        # Step 1: Speech-to-text with Whisper
        whisper_result = self._whisper_model.transcribe(audio_path)
        segments = whisper_result.get("segments", [])

        # Step 2: Speaker diarization with Pyannote
        diarization = self._diarization_pipeline(audio_path)

        # Step 3: Align Whisper segments with Pyannote speakers
        lines: list[str] = []
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()

            speaker = self._find_speaker(diarization, start, end)
            timestamp = self._format_timestamp(start)
            lines.append(f"[{timestamp}] {speaker}: {text}")

        transcript = "\n".join(lines)
        logger.info(
            "transcription_complete",
            audio_path=audio_path,
            segments=len(segments),
            lines=len(lines),
        )
        return transcript

    @staticmethod
    def _find_speaker(diarization, start: float, end: float) -> str:
        """Find the speaker for a time segment using maximum temporal overlap."""
        best_speaker = "Unknown"
        best_overlap = 0.0

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(turn.start, start)
            overlap_end = min(turn.end, end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        return best_speaker

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
