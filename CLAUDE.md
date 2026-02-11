# Meeting Intelligence System

## Project Summary
Meeting Intelligence System — a conversational AI assistant that analyses meeting transcripts
(text files with speaker labels and timestamps) and answers questions about discussions,
decisions, and action items. Built as a take-home assignment for an AI Engineer role at
Newpage Solutions (Bristol, UK).

## Tech Stack
| Component | Technology |
|---|---|
| Language | Python 3.11+ (type hints everywhere) |
| LLM | Google Gemini 2.5 Flash (free tier) — primary; Claude Sonnet + Ollama as alternatives |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local, 384-dim) |
| Vector DB | Qdrant (Docker service, cosine similarity) |
| Structured DB | SQLite (action items, decisions, summaries) |
| API | FastAPI (async, OpenAPI docs) |
| Frontend | Streamlit (demo UI) |
| Orchestration | Custom Python chain pattern (no LangChain) |
| Containers | Docker + docker-compose (app + qdrant) |
| Logging | structlog (structured JSON) |
| Testing | pytest |

## Coding Standards
- Python 3.11+, type hints on ALL function parameters and return types
- Pydantic models for ALL data structures crossing module boundaries
- No magic strings — use enums and constants
- Dependency injection — components receive dependencies, don't create them
- Error handling — wrap external calls (Qdrant, LLM APIs) with try/except
- Logging — structlog on every significant operation
- Docstrings — on all public classes and methods
- Small, focused functions — each function does one thing
- No print statements — use logger
- Consistent naming — snake_case for functions/variables, PascalCase for classes
- async where appropriate (LLM calls, API routes)

## Key Architecture Decisions
1. **Custom orchestration** — No LangChain. Explicit, testable Python chain pattern.
2. **Dual storage** — Qdrant for vector search + SQLite for structured data (action items, decisions).
3. **Speaker-aware chunking** — Split on speaker change, merge short turns, split long monologues.
4. **Provider abstraction** — Abstract LLMProvider base class; swap Gemini/Claude/Ollama via config.
5. **Intent classification** — Route queries to appropriate retrieval strategy (structured, speaker, semantic, cross-meeting).

## Project Structure
```
meeting-intelligence/
├── CLAUDE.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── Makefile
├── docs/ (architecture.md, adr/)
├── data/sample_transcripts/
├── src/ (config, models, ingestion, embeddings, storage, retrieval, llm, chains, prompts, api, voice)
├── ui/ (streamlit_app.py)
└── tests/
```

## Testing
- pytest for core logic: parser, chunker, retriever, query chain
- Mock LLM calls in tests
- Test both transcript formats (timestamped + simple labels)
