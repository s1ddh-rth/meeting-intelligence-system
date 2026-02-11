# Architecture Overview

## System Architecture

The Meeting Intelligence System is built around two core pipelines — **Ingestion** and **Query** — connected by a dual storage layer (Qdrant + SQLite).

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
│                                                                     │
│  ┌───────────────┐    ┌───────────────────────────────────────────────┐  │
│  │  Streamlit UI  │───▶│              FastAPI (port 8000)              │  │
│  │  (port 8501)   │    │  /api/ingest  /api/ingest/audio  /api/query  │  │
│  └───────────────┘    └──────────────────────┬────────────────────────┘  │
└──────────────────────────────────────┼──────────────────────────────┘
                                       │
┌──────────────────────────────────────┼──────────────────────────────┐
│                      ORCHESTRATION LAYER                            │
│                                       │                             │
│  ┌────────────────────┐   ┌──────────┴──────────┐                  │
│  │  Ingestion Chain    │   │   Query Chain        │                  │
│  │                     │   │                      │                  │
│  │  [transcribe] →     │   │  classify intent →   │                  │
│  │  parse → chunk →    │   │  [speaker check] →   │                  │
│  │  embed → store →    │   │  retrieve context →  │                  │
│  │  extract → store    │   │  generate answer     │                  │
│  └────────┬───────────┘   └──────────┬───────────┘                  │
└───────────┼──────────────────────────┼──────────────────────────────┘
            │                          │
┌───────────┼──────────────────────────┼──────────────────────────────┐
│           │       COMPONENT LAYER    │                              │
│           │                          │                              │
│  ┌────────┴────────┐   ┌────────────┴────────┐   ┌──────────────┐ │
│  │AudioTranscriber  │   │    Retriever         │   │ LLM Provider │ │
│  │ TranscriptParser │   │  (vector + struct)   │   │ (Gemini /    │ │
│  │ SpeakerChunker   │   │                      │   │  Claude /    │ │
│  │ Embedder         │   │                      │   │  Ollama)     │ │
│  │ Extractor        │   │                      │   │              │ │
│  └─────────────────┘   └──────────────────────┘   └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
            │                          │
┌───────────┼──────────────────────────┼──────────────────────────────┐
│           │        STORAGE LAYER     │                              │
│           ▼                          ▼                              │
│  ┌─────────────────┐   ┌──────────────────────┐                    │
│  │   Qdrant         │   │   SQLite              │                    │
│  │   (Docker)       │   │   (embedded)          │                    │
│  │                  │   │                       │                    │
│  │  Vector chunks   │   │  Action items         │                    │
│  │  + metadata      │   │  Decisions            │                    │
│  │  (384-dim cosine)│   │  Summaries            │                    │
│  │                  │   │  Meeting metadata      │                    │
│  └─────────────────┘   └──────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Ingestion Pipeline

```
Audio File (.mp3/.wav/.m4a)          Transcript File (.txt)
    │                                     │
    ▼                                     │
┌──────────────────┐                      │
│ AudioTranscriber  │  faster-whisper +    │
│                   │  pyannote diarize    │
│                   │  → save .txt file    │
└────────┬─────────┘                      │
         │ .txt transcript                │
         └──────────┬─────────────────────┘
                    ▼
┌──────────────────┐
│ TranscriptParser  │  Auto-detect format (timestamped or simple labels)
│                   │  Parse into Utterance objects
└────────┬─────────┘
         │ list[Utterance]
         ▼
┌──────────────────┐
│ SpeakerChunker    │  Split on speaker changes
│                   │  Merge short turns, split long monologues
│                   │  Attach rich metadata
└────────┬─────────┘
         │ list[Chunk]
         ▼
┌──────────────────┐
│ Embedder          │  sentence-transformers/all-MiniLM-L6-v2
│                   │  Batch encode chunks → 384-dim vectors
└────────┬─────────┘
         │ list[list[float]]
         ▼
┌──────────────────┐     ┌──────────────────┐
│ VectorStore       │     │ StructuredStore   │
│ (Qdrant)          │     │ (SQLite)          │
│                   │     │                   │
│ Chunks + vectors  │     │ Action items      │
│ + metadata        │     │ Decisions         │
└──────────────────┘     │ Topics, Summary   │
                          └──────────────────┘
                                   ▲
                                   │
                          ┌────────┴─────────┐
                          │ StructuredExtract  │  LLM extracts structured
                          │ (LLM-powered)      │  data from full transcript
                          └──────────────────┘
```

### Query Pipeline

```
User Question
    │
    ▼
┌──────────────────┐
│ Intent Classifier │  LLM classifies: STRUCTURED | SPEAKER | SEMANTIC | CROSS_MEETING
│                   │  Falls back to heuristic keywords if LLM is rate-limited
└────────┬─────────┘
         │ QueryIntent
         ▼
┌──────────────────┐     ┌─────────────────────────────────┐
│ Speaker Check     │────▶│ Speaker not found?               │
│ (SPEAKER intent)  │     │ → Return definitive answer with  │
│                   │     │   actual speaker list (no LLM)   │
└────────┬─────────┘     └─────────────────────────────────┘
         │ speaker exists (or non-speaker intent)
         ▼
┌──────────────────┐
│ Retriever         │  Routes based on intent:
│                   │  - STRUCTURED → SQLite (action items, decisions)
│                   │  - SPEAKER → Qdrant with speaker filter
│                   │  - SEMANTIC → Qdrant similarity search
│                   │  - CROSS_MEETING → Qdrant (no meeting filter)
└────────┬─────────┘
         │ RetrievalResult (chunks + structured context)
         ▼
┌──────────────────┐
│ Confidence Scorer │  Compute confidence from retrieval quality:
│                   │  HIGH (>= 0.7) | MEDIUM (>= 0.4) | LOW (< 0.4)
└────────┬─────────┘
         │ (empty results → honest "no info" answer, skip LLM)
         ▼
┌──────────────────┐
│ Prompt Builder    │  Inject retrieved context into system prompt
└────────┬─────────┘
         │ formatted prompt
         ▼
┌──────────────────┐
│ LLM Provider      │  Generate grounded answer with citations
│ (Gemini/Claude/   │  (rate-limit fallback: return raw excerpts)
│  Ollama)          │
└────────┬─────────┘
         │ LLMResponse
         ▼
┌──────────────────┐
│ QueryResponse     │  Answer + sources + intent + confidence + latency
└──────────────────┘
```

## Docker Services

```
┌─────────────────────────────────┐     ┌─────────────────────┐
│          app container          │     │   qdrant container   │
│                                 │     │                      │
│  FastAPI  (:8000)               │────▶│  Qdrant  (:6333)     │
│  Streamlit (:8501)              │     │                      │
│  Embedder (in-process)          │     │  Vector storage      │
│  SQLite (file-based)            │     │  (Rust engine)       │
│                                 │     │                      │
│  Volumes:                       │     │  Volumes:            │
│    ./data → /app/data           │     │    qdrant-data       │
│    sqlite-data → /app/data/sqlite│    │                      │
└─────────────────────────────────┘     └─────────────────────┘
```

## Key Design Principles

1. **Separation of concerns**: Each module has a single responsibility
2. **Dependency injection**: Components receive dependencies, don't create them
3. **Dual storage**: Vector search for semantic queries, SQLite for structured queries
4. **Provider abstraction**: LLM provider is swappable via configuration
5. **Speaker awareness**: Chunking, metadata, and filtering preserve conversation structure
6. **Idempotent ingestion**: Re-ingesting a transcript replaces existing data cleanly
7. **Graceful degradation**: Speaker existence checks, confidence scoring, and heuristic classification work even when the LLM is rate-limited
8. **Lazy model loading**: Audio transcription models (whisper + pyannote) load only on first use — text-only users pay no cost
