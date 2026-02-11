"""FastAPI routes for the Meeting Intelligence API."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel

from src.llm.provider import RateLimitError
from src.models.schemas import (
    ActionItem,
    AudioIngestionResult,
    Decision,
    IngestionResult,
    MeetingInfo,
    QueryResponse,
)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}

logger = structlog.get_logger()

router = APIRouter()


# --- Request/Response models ---


class QueryRequest(BaseModel):
    """Request body for the query endpoint."""

    question: str
    meeting_id: str | None = None


class HealthResponse(BaseModel):
    """Response for the health check endpoint."""

    status: str
    qdrant_connected: bool
    llm_provider: str


# --- Routes ---


@router.post("/ingest", response_model=IngestionResult)
async def ingest_transcript(request: Request, file: UploadFile) -> IngestionResult:
    """Upload and process a meeting transcript.

    Parses the transcript, creates speaker-aware chunks, generates embeddings,
    stores vectors in Qdrant, and extracts structured data to SQLite.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    content = await file.read()
    text = content.decode("utf-8")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Transcript file is empty")

    logger.info("ingest_request", filename=file.filename, size_bytes=len(content))

    try:
        chain = request.app.state.ingestion_chain
        result = await chain.run(file_content=text, filename=file.filename)
        return result
    except Exception:
        logger.exception("ingest_failed", filename=file.filename)
        raise HTTPException(status_code=500, detail="Ingestion failed — check logs for details")


@router.post("/ingest/audio", response_model=AudioIngestionResult)
async def ingest_audio(request: Request, file: UploadFile) -> AudioIngestionResult:
    """Upload an audio file, transcribe it with speaker labels, and ingest.

    Accepts .mp3, .wav, .m4a, .ogg, .flac, .webm files. The audio is
    transcribed using faster-whisper + pyannote speaker diarization, saved
    as a .txt transcript in data/transcripts/, then fed into the standard
    ingestion pipeline.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Accepted: {', '.join(sorted(AUDIO_EXTENSIONS))}",
        )

    transcriber = getattr(request.app.state, "transcriber", None)
    if transcriber is None:
        raise HTTPException(
            status_code=501,
            detail="Audio transcription is not available. Install voice dependencies: pip install -r requirements-voice.txt",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    logger.info("audio_ingest_request", filename=file.filename, size_bytes=len(content))

    # Write audio to temp file for processing
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Transcribe audio → text
        transcript_text = transcriber.transcribe(tmp_path)
    except (ImportError, ValueError) as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception:
        logger.exception("audio_transcription_failed", filename=file.filename)
        raise HTTPException(status_code=500, detail="Audio transcription failed — check logs for details")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not transcript_text.strip():
        raise HTTPException(status_code=422, detail="Transcription produced no text — audio may be silent or corrupted")

    # Save transcript to data/transcripts/
    settings = request.app.state.settings
    transcripts_dir = Path(settings.transcripts_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    transcript_filename = Path(file.filename).stem + ".txt"
    transcript_path = transcripts_dir / transcript_filename
    transcript_path.write_text(transcript_text, encoding="utf-8")
    logger.info("transcript_saved", path=str(transcript_path))

    # Feed into existing ingestion pipeline
    try:
        chain = request.app.state.ingestion_chain
        result = await chain.run(file_content=transcript_text, filename=transcript_filename)
    except Exception:
        logger.exception("ingestion_after_transcription_failed", filename=transcript_filename)
        raise HTTPException(status_code=500, detail="Ingestion failed after transcription — check logs for details")

    return AudioIngestionResult(
        meeting_id=result.meeting_id,
        chunks_created=result.chunks_created,
        speakers=result.speakers,
        topics=result.topics,
        action_items_count=result.action_items_count,
        transcript_filename=transcript_filename,
        transcript_text=transcript_text,
    )


@router.post("/query", response_model=QueryResponse)
async def query_meetings(request: Request, body: QueryRequest) -> QueryResponse:
    """Ask a question about meeting transcripts.

    Classifies the query intent, retrieves relevant context, and generates
    a grounded answer with source citations.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(
        "query_request",
        question_preview=body.question[:80],
        meeting_id=body.meeting_id,
    )

    try:
        chain = request.app.state.query_chain
        result = await chain.run(query=body.question, meeting_id=body.meeting_id)
        return result
    except RateLimitError as e:
        logger.warning("query_rate_limited", question=body.question[:80])
        raise HTTPException(
            status_code=429,
            detail=str(e),
        )
    except Exception:
        logger.exception("query_failed", question=body.question[:80])
        raise HTTPException(status_code=500, detail="Query processing failed — check logs for details")


@router.get("/meetings", response_model=list[MeetingInfo])
async def list_meetings(request: Request) -> list[MeetingInfo]:
    """List all ingested meetings."""
    store = request.app.state.structured_store
    return store.list_meetings()


@router.get("/meetings/{meeting_id}", response_model=MeetingInfo)
async def get_meeting(request: Request, meeting_id: str) -> MeetingInfo:
    """Get details for a specific meeting."""
    store = request.app.state.structured_store
    meeting = store.get_meeting_details(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id}' not found")
    return meeting


@router.get("/meetings/{meeting_id}/action-items", response_model=list[ActionItem])
async def get_action_items(request: Request, meeting_id: str) -> list[ActionItem]:
    """Get action items for a specific meeting."""
    store = request.app.state.structured_store
    return store.get_action_items(meeting_id)


@router.get("/meetings/{meeting_id}/decisions", response_model=list[Decision])
async def get_decisions(request: Request, meeting_id: str) -> list[Decision]:
    """Get decisions for a specific meeting."""
    store = request.app.state.structured_store
    return store.get_decisions(meeting_id)


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Health check — verifies Qdrant connection and reports LLM provider."""
    vector_store = request.app.state.vector_store
    settings = request.app.state.settings
    qdrant_ok = vector_store.health_check()

    status = "healthy" if qdrant_ok else "degraded"
    return HealthResponse(
        status=status,
        qdrant_connected=qdrant_ok,
        llm_provider=settings.llm_provider,
    )
