"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI

from src.api.routes import router
from src.chains.ingestion_chain import IngestionChain
from src.chains.query_chain import QueryChain
from src.config.settings import Settings
from src.embeddings.embedder import Embedder
from src.ingestion.chunker import SpeakerAwareChunker
from src.ingestion.extractor import StructuredExtractor
from src.ingestion.parser import TranscriptParser
from src.llm.factory import get_llm_provider
from src.retrieval.retriever import Retriever
from src.storage.structured_store import StructuredStore
from src.storage.vector_store import VectorStore
from src.voice.transcriber import AudioTranscriber

logger = structlog.get_logger()

# Configure structlog for JSON output
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler — initialise all components on startup."""
    settings = Settings()

    # Core components
    embedder = Embedder(model_name=settings.embedding_model)
    vector_store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection,
        embedding_dimension=settings.embedding_dimension,
    )
    structured_store = StructuredStore(db_path=settings.sqlite_path)
    llm = get_llm_provider(settings)

    # Initialise stores
    vector_store.init_collection()
    structured_store.init_db()

    # Pipeline components
    parser = TranscriptParser()
    chunker = SpeakerAwareChunker(
        max_chunk_tokens=settings.max_chunk_tokens,
        min_chunk_tokens=settings.min_chunk_tokens,
    )
    extractor = StructuredExtractor(llm=llm)

    # Chains
    ingestion_chain = IngestionChain(
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        structured_store=structured_store,
        extractor=extractor,
    )
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        structured_store=structured_store,
        top_k=settings.top_k,
        similarity_threshold=settings.similarity_threshold,
    )
    query_chain = QueryChain(retriever=retriever, llm=llm)

    # Voice transcriber (lazy-loaded — models download on first audio upload)
    transcriber = AudioTranscriber(
        whisper_model=settings.whisper_model,
        hf_token=settings.hf_token,
    )

    # Attach to app state for dependency injection in routes
    app.state.settings = settings
    app.state.ingestion_chain = ingestion_chain
    app.state.query_chain = query_chain
    app.state.structured_store = structured_store
    app.state.vector_store = vector_store
    app.state.transcriber = transcriber

    logger.info(
        "app_started",
        llm_provider=settings.llm_provider,
        qdrant_host=settings.qdrant_host,
        embedding_model=settings.embedding_model,
    )

    yield

    logger.info("app_shutdown")


app = FastAPI(
    title="Meeting Intelligence System",
    description="AI-powered meeting transcript analysis with RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api")
