"""FastAPI routes for the Meeting Intelligence API."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request, UploadFile
from pydantic import BaseModel

from src.llm.provider import RateLimitError
from src.models.schemas import (
    ActionItem,
    Decision,
    IngestionResult,
    MeetingInfo,
    QueryResponse,
)

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
