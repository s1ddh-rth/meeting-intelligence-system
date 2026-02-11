# Architecture Overview

## System Architecture

The Meeting Intelligence System is built around two core pipelines — **Ingestion** and **Query** — connected by a dual storage layer (Qdrant + SQLite).

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
│                                                                     │
│  ┌───────────────┐    ┌──────────────────────────────────────┐     │
│  │  Streamlit UI  │───▶│         FastAPI (port 8000)          │     │
│  │  (port 8501)   │    │  /api/ingest  /api/query  /api/...  │     │
│  └───────────────┘    └──────────────┬───────────────────────┘     │
└──────────────────────────────────────┼──────────────────────────────┘
                                       │
┌──────────────────────────────────────┼──────────────────────────────┐
│                      ORCHESTRATION LAYER                            │
│                                       │                             │
│  ┌────────────────────┐   ┌──────────┴──────────┐                  │
│  │  Ingestion Chain    │   │   Query Chain        │                  │
│  │                     │   │                      │                  │
│  │  parse → chunk →    │   │  classify intent →   │                  │
│  │  embed → store →    │   │  retrieve context →  │                  │
│  │  extract → store    │   │  generate answer     │                  │
│  └────────┬───────────┘   └──────────┬───────────┘                  │
└───────────┼──────────────────────────┼──────────────────────────────┘
            │                          │
┌───────────┼──────────────────────────┼──────────────────────────────┐
│           │       COMPONENT LAYER    │                              │
│           │                          │                              │
│  ┌────────┴────────┐   ┌────────────┴────────┐   ┌──────────────┐ │
│  │ TranscriptParser │   │    Retriever         │   │ LLM Provider │ │
│  │ SpeakerChunker   │   │  (vector + struct)   │   │ (Gemini /    │ │
│  │ Embedder         │   │                      │   │  Claude /    │ │
│  │ Extractor        │   │                      │   │  Ollama)     │ │
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
Transcript File (.txt)
    │
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
         ▲                └──────────────────┘
         │                         ▲
         │                         │
┌────────┴─────────┐    ┌─────────┴────────┐
│ Embedder          │    │ StructuredExtract │  LLM extracts structured
│ (batch encode)    │    │ (LLM-powered)     │  data from full transcript
└──────────────────┘    └──────────────────┘
```

### Query Pipeline

```
User Question
    │
    ▼
┌──────────────────┐
│ Intent Classifier │  LLM classifies: STRUCTURED | SPEAKER | SEMANTIC | CROSS_MEETING
└────────┬─────────┘
         │ QueryIntent
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
│ Prompt Builder    │  Inject retrieved context into system prompt
└────────┬─────────┘
         │ formatted prompt
         ▼
┌──────────────────┐
│ LLM Provider      │  Generate grounded answer with citations
│ (Gemini/Claude/   │
│  Ollama)          │
└────────┬─────────┘
         │ LLMResponse
         ▼
┌──────────────────┐
│ QueryResponse     │  Answer + sources + intent + latency metrics
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
