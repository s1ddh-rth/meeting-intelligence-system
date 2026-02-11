"""Audio transcription using faster-whisper + pyannote for speaker diarization.

BONUS MODULE — Optional. Requires additional dependencies:
    pip install -r requirements-voice.txt

This module converts audio files (.wav, .mp3, .m4a) into timestamped,
speaker-labeled transcripts compatible with the existing parser pipeline (Format A).

Uses faster-whisper (CTranslate2-based, INT8, CPU) instead of openai-whisper
for ~4x faster inference and lower memory usage.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()


class AudioTranscriber:
    """Converts audio to speaker-labelled transcript using faster-whisper + pyannote.

    Combines faster-whisper (speech-to-text) with pyannote (speaker diarization)
    by aligning whisper segments with pyannote speaker labels via temporal overlap.

    The output is a formatted transcript string matching Format A (timestamped),
    which feeds directly into the existing TranscriptParser.

    Models are lazy-loaded on first call to transcribe() so text-only users
    pay no startup cost.
    """

    def __init__(self, whisper_model: str = "base", hf_token: str = "") -> None:
        self._whisper_model_name = whisper_model
        self._hf_token = hf_token
        self._whisper_model: Any = None
        self._diarization_pipeline: Any = None

    def _load_models(self) -> None:
        """Lazy-load faster-whisper and pyannote models on first use."""
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel

                logger.info(
                    "loading_whisper_model",
                    model=self._whisper_model_name,
                    compute_type="int8",
                )
                self._whisper_model = WhisperModel(
                    self._whisper_model_name,
                    device="cpu",
                    compute_type="int8",
                )
            except ImportError:
                raise ImportError(
                    "faster-whisper is required for audio transcription. "
                    "Install with: pip install -r requirements-voice.txt"
                )

        if self._diarization_pipeline is None:
            if not self._hf_token:
                raise ValueError(
                    "HF_TOKEN is required for pyannote speaker diarization. "
                    "Get a free token at https://huggingface.co/settings/tokens "
                    "and accept the model terms at "
                    "https://huggingface.co/pyannote/speaker-diarization-3.1"
                )

            try:
                from pyannote.audio import Pipeline

                logger.info("loading_pyannote_pipeline")
                self._diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self._hf_token,
                )
            except ImportError:
                raise ImportError(
                    "pyannote.audio is required for speaker diarization. "
                    "Install with: pip install -r requirements-voice.txt"
                )

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to a speaker-labelled transcript.

        Args:
            audio_path: Path to .wav, .mp3, or .m4a file.

        Returns:
            Formatted transcript string (Format A: [HH:MM:SS] Speaker N: text).
        """
        self._load_models()

        logger.info("transcription_started", audio_path=audio_path)

        # Step 1: Speech-to-text with faster-whisper
        segments_iter, info = self._whisper_model.transcribe(
            audio_path,
            beam_size=5,
        )
        segments = list(segments_iter)
        logger.info(
            "whisper_complete",
            language=info.language,
            language_probability=round(info.language_probability, 2),
            num_segments=len(segments),
        )

        # Step 2: Speaker diarization with pyannote
        # Load audio via torchaudio and resample to 16kHz — pyannote expects
        # this sample rate and fails on raw MP3s with non-standard rates.
        import torch
        import torchaudio

        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        # Pyannote expects mono; mix down if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        pipeline_output = self._diarization_pipeline(
            {"waveform": waveform, "sample_rate": 16000}
        )

        # pyannote 4.x returns DiarizeOutput; extract the Annotation object
        diarization = getattr(pipeline_output, "speaker_diarization", pipeline_output)

        # Build speaker rename map: SPEAKER_00 → Speaker 1, SPEAKER_01 → Speaker 2
        raw_labels = sorted({label for _, _, label in diarization.itertracks(yield_label=True)})
        speaker_map = {
            raw: f"Speaker {i + 1}"
            for i, raw in enumerate(raw_labels)
        }
        logger.info("diarization_complete", speakers=list(speaker_map.values()))

        # Step 3: Align whisper segments with pyannote speakers
        lines: list[str] = []
        for segment in segments:
            start = segment.start
            end = segment.end
            text = segment.text.strip()
            if not text:
                continue

            raw_speaker = self._find_speaker(diarization, start, end)
            speaker = speaker_map.get(raw_speaker, "Unknown")
            timestamp = self._format_timestamp(start)
            lines.append(f"[{timestamp}] {speaker}: {text}")

        transcript = "\n".join(lines)
        logger.info(
            "transcription_complete",
            audio_path=audio_path,
            num_segments=len(segments),
            num_lines=len(lines),
            num_speakers=len(speaker_map),
        )
        return transcript

    @staticmethod
    def _find_speaker(diarization: Any, start: float, end: float) -> str:
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
