# Meeting Intelligence System

AI-powered meeting transcript analysis with RAG — ask questions about discussions, decisions, and action items.

## Quick Setup

### Prerequisites
- Docker and Docker Compose
- A Gemini API key ([get one free](https://aistudio.google.com/apikey))

### 1. Clone and configure
```bash
git clone <repo-url>
cd meeting-intelligence
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Run with Docker
```bash
make run
# or: docker-compose up --build
```

### 3. Access the application
- **Streamlit UI**: http://localhost:8501
- **FastAPI docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/api/health

### 4. Try it out
1. Open the Streamlit UI
2. Upload a sample transcript from `data/sample_transcripts/`
3. Ask questions like:
   - "What are the action items?"
   - "What did Sarah say about the launch?"
   - "What decisions were made?"
   - "Summarise the meeting"

## Architecture Overview
<!-- Link to docs/architecture.md, high-level diagram -->
See [docs/architecture.md](docs/architecture.md) for detailed architecture diagrams and data flow.

The system has two core pipelines:
- **Ingestion**: Parse transcript → Speaker-aware chunking → Embed → Store in Qdrant + Extract structured data to SQLite
- **Query**: Classify intent → Retrieve context (vector or structured) → Generate grounded answer with citations

## RAG/LLM Approach & Decisions
<!-- Link to ADRs, summary of key choices -->
All architectural decisions are documented as ADRs in [docs/adr/](docs/adr/):

| Decision | ADR |
|---|---|
| Vector database (Qdrant) | [ADR-001](docs/adr/001-vector-database-selection.md) |
| Embedding model (MiniLM) | [ADR-002](docs/adr/002-embedding-model-selection.md) |
| LLM provider (Gemini + abstraction) | [ADR-003](docs/adr/003-llm-provider-selection.md) |
| Chunking strategy (speaker-aware) | [ADR-004](docs/adr/004-chunking-strategy.md) |
| Dual storage (Qdrant + SQLite) | [ADR-005](docs/adr/005-dual-storage-architecture.md) |
| Orchestration (custom, no LangChain) | [ADR-006](docs/adr/006-orchestration-approach.md) |

## Key Technical Decisions
<!-- PLACEHOLDER — I will write this section myself -->

## Engineering Standards
<!-- PLACEHOLDER — I will write this section myself -->

## How I Used AI Tools in Development
<!-- PLACEHOLDER — I will write this section myself -->

## Productionising & Cloud Deployment
<!-- PLACEHOLDER — I will write this section myself -->

## Known Limitations & Production Considerations

### Structured Query Latency
Structured queries (e.g. "list action items", "who attended?") currently take 6-7 seconds because they still pass through the LLM for intent classification and answer generation, even though the data comes directly from SQLite. Semantic queries (2-3s) are actually faster because they skip the structured retrieval overhead.

**Production optimisation:** For simple list/lookup queries (action items, decisions, speakers, summaries), the system should bypass the LLM entirely and return formatted SQLite results directly. Intent classification could use a lightweight local model or deterministic keyword matching, and the structured data could be returned as-is without LLM reformatting. This would bring structured query latency down to <100ms.

### Embedding Similarity Scores
Vector search scores may appear low (< 0.5) for some queries. This is expected behaviour with `all-MiniLM-L6-v2` embeddings and cosine similarity — conversational transcript text often has weak semantic overlap with formal question phrasing. The system now falls back to returning top-3 results with a low-confidence flag when no chunks pass the similarity threshold, rather than returning empty results.

## What I'd Do Differently With More Time
<!-- PLACEHOLDER — I will write this section myself -->

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/ingest` | POST | Upload and process a transcript |
| `/api/query` | POST | Ask a question about meetings |
| `/api/meetings` | GET | List all ingested meetings |
| `/api/meetings/{id}` | GET | Get meeting details |
| `/api/meetings/{id}/action-items` | GET | Get action items |
| `/api/meetings/{id}/decisions` | GET | Get decisions |
| `/api/health` | GET | Health check |

Full interactive docs at http://localhost:8000/docs when running.

## Running Tests
```bash
make test
# or: pytest tests/ -v
```

## Project Structure
```
meeting-intelligence/
├── src/
│   ├── config/          # Pydantic settings
│   ├── models/          # Data models (schemas)
│   ├── ingestion/       # Parser, chunker, extractor
│   ├── embeddings/      # Sentence-transformer wrapper
│   ├── storage/         # Qdrant + SQLite stores
│   ├── retrieval/       # Combined retriever
│   ├── llm/             # Provider abstraction (Gemini, Claude, Ollama)
│   ├── chains/          # Ingestion + query orchestration
│   ├── prompts/         # All LLM prompts
│   └── api/             # FastAPI routes
├── ui/                  # Streamlit frontend
├── tests/               # pytest suite
├── data/                # Sample transcripts
└── docs/                # Architecture + ADRs
```
