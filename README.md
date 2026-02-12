# Meeting Intelligence System

AI-powered meeting transcript analysis with RAG — ask questions about discussions, decisions, and action items.

## Quick Setup

### Prerequisites
- Docker and Docker Compose (or Podman + podman-compose)
- A Gemini API key ([get one free](https://aistudio.google.com/apikey))
- *(Optional, for audio ingestion)* A HuggingFace token ([get one free](https://huggingface.co/settings/tokens)) with access to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### 1. Clone and configure
```bash
git clone <repo-url>
cd meeting-intelligence-system
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
# (Optional) Add HF_TOKEN for audio transcription
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
2. Upload a transcript (`.txt`) or audio file (`.mp3`, `.wav`, `.m4a`), or choose an existing meeting from the `Meetings` dropdown
3. Ask questions like:
   - "What are the action items?"
   - "What did Sarah say about the launch?"
   - "What decisions were made?"
   - "Summarise the meeting"

## Architecture Overview
<!-- Link to docs/architecture.md, high-level diagram -->
See [docs/architecture.md](docs/architecture.md) for detailed architecture diagrams and data flow.

The system has two core pipelines:
- **Ingestion**: Parse transcript → Speaker-aware chunking → Embed → Store in Qdrant + Extract structured data to SQLite. Audio files are first transcribed with speaker diarization (faster-whisper + pyannote) before entering the same pipeline.
- **Query**: Classify intent → Retrieve context (vector or structured) → Generate grounded answer with confidence scores and citations

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
**Speaker-aware chunking over fixed-size chunking:** The whole point is that meetings are conversational, not documents. Fixed-size chunking breaks sentence midway, which leads to losing speaker context and make it impossible to filter by who said what. Speaker-aware chunking preserves conversational structure and enables metadata filtering in Qdrant.

**Dual Storage - Qdrant + SQLite:** Not every question needs vector search. Something like "List action items" is a structured lookup, not a semantic similarity problem. Pre-extracting structured data at ingestion time means structured queries work even when the LLM is rate-limited or unavailable. 

**Custom orchestration:** For a focused RAG system with two clear pipelines (ingest and query), the complexity of a framework didn't feel justified. The chains are explicit, testable and debuggable. This system follows the same retrieve-augment-generate pattern just without the abstraction layer. For more complex workflows I'd evaluate LangGraph.

**Provider abstraction for LLMs:** In this system, I didn't hardcode any specific provider. The abstract base class means that swapping to Claude or Ollama is a config change. This totally avoids vendor lock-in and lets the user try out different providers.

**Intent classification:** Not all queries should take the same path through the system. Speaker queries filter by metadata, structured queries hit SQLite, semantic queries do full vector search. It's efficient and gives better results than running everything through the same retrieval pipeline.

**Speaker existence fast-path:** For speaker queries, the system checks the known speakers list in SQLite before any vector search. If the speaker doesn't exist, it returns an immediate definitive answer listing the actual speakers - no vector search or LLM call needed.

**Confidence scoring:** Every answer includes a confidence level (High / Medium / Low) displayed as a colour-coded badge in the UI. Structured queries from SQLite are always HIGH. For vector search results, confidence is based on the top similarity score (>= 0.7 HIGH, >= 0.4 MEDIUM, < 0.4 LOW).

## Engineering Standards
**What was followed:** Type hints everywhere, Pydantic models for all data crossing module boundaries, dependency injection (components receive dependencies rather than creating them), structured logging with structlog (JSON format, every significant operation logged with context), separation of concerns (parser doesn't know about embeddings, embedder doesn't know about Qdrant), configuration via environment variables with Pydantic BaseSettings.

**Testing approach:** pytest covering core logic - parser format detection, chunker boundary logic, retriever filtering. 

**What was consciously skipped and why:** Authentication (didn't feel relevant for a demo, would add OAuth2/API keys in production), comprehensive error handling (basic try/except with logging, production would add circuit breakers and retry policies), full-fledged CI/CD pipeline (GitHub Actions runs lint + tests on every push; deployment is manual since self-hosted runners are unsafe on public repos), input validation beyond Pydantic (would add file size limits, content type checking, rate limiting).

## How I Used AI Tools in Development
**Our workflow:** Used Claude, and ChatGPT for architecture discussion, and technology comparison. Used Claude Code CLI for implementation with a CLAUDE.md context file that defined the project structure, coding standards, and architectural constraints.

I'd generally go for setting up a proposal for the whole project with [openspec](https://openspec.dev/) and combine it with Ralph loop to go on about execution but the scope here didn't warrant it.

**The division of labour:** I made all architectural decisions - tech stack selection, chunking strategy, dual storage design, provider abstraction pattern. I designed the data models and pipeline flow. Claude Code implemented modules under my direction, following the standards in CLAUDE.md. I reviewed all generated code, tested it, and iterated on issues I found.

**What CLAUDE.md gave me:** Consistent output across sessions. Without it, each Claude Code session would need re-explaining the project context. With it, the AI had persistent awareness of the architecture, coding standards, and project structure. This made AI-assisted development repeatable rather than ad-hoc.

## Productionising & Cloud Deployment
**Database upgrades:** Qdrant stays (it's already production-grade) but moves to a managed instance or dedicated cluster. From SQLite → PostgreSQL on RDS/Cloud SQL for durability, concurrent access, and proper backups.

**Compute and scaling:** Docker Compose → ECS/Cloud Run/Kubernetes. Separate the API and Streamlit into independent services. Add a load balancer. The ingestion pipeline becomes async - upload triggers a background job (Celery + Redis or SQS) so the user doesn't wait for processing.

**LLM provider:** I'd move from Gemini free tier to Claude or GPT-4o with proper API contracts and rate limits. Add response caching (Redis) for repeated queries. Can also consider a smaller local model for intent classification to reduce API calls.

**Observability:** Structured logs → shipped to CloudWatch/Datadog. Add distributed tracing for the full query pipeline. Track metrics: query latency p50/p95/p99, retrieval relevance scores, LLM token usage, error rates. Set up alerts for degraded retrieval quality.

**Security:** API authentication (OAuth2 or API keys), input sanitisation, file upload limits, PII detection in transcripts (critical for a healthcare company), data encryption at rest and in transit.

**CI/CD:** GitHub Actions pipeline - lint (ruff), test (pytest), build Docker image, push to registry, deploy to staging, run integration tests, promote to production.

## Known Limitations & Considerations

### Structured Query Latency
The dual storage design is good. However, on the happy path, structured queries currently take 6-7 seconds because they still route through the LLM for intent classification and answer generation. Semantic queries (2-3s) are actually faster because they skip the structured retrieval overhead.

**Production optimisation:** For simple list/lookup queries (action items, decisions, speakers, summaries), the system should bypass the LLM entirely and return formatted SQLite results directly. Intent classification could use a lightweight local model or deterministic keyword matching, and the structured data could be returned as-is without LLM reformatting. This would bring structured query latency down to <100ms.

### Heuristic Classifier Degradation
When the LLM is rate-limited, intent classification falls back to keyword heuristics. This fallback can miss speaker queries phrased indirectly (e.g. "was sarah johnson here though?"), causing them to fall through to semantic search instead of the speaker existence fast-path described above. This means the definitive "speaker not found" answer may not trigger for unusual phrasings under rate-limiting.

### Embedding Similarity Scores
Vector search scores may appear low (< 0.5) for some queries. This is expected behaviour with `all-MiniLM-L6-v2` embeddings and cosine similarity - it just so happens that conversational transcript text often has weak semantic overlap with formal question phrasing. The system now falls back to returning top-3 results with a low-confidence flag when no chunks pass the similarity threshold, rather than returning empty results.

## What I'd Do Differently With More Time
**Retrieval quality:** Evaluate HyDE (Hypothetical Document Embedding) for vague queries. Would definitely add re-ranking with a cross-encoder model after initial retrieval. Experiment with hybrid search (BM25 keyword matching + vector similarity). Fine-tune the similarity threshold based on evaluation data rather than a hardcoded 0.3.

**Evaluation framework:** Build a test suite of question-answer pairs with expected sources. Measure retrieval recall, answer accuracy, and hallucination rate systematically rather than manual testing.

**Chunking improvements:** Experiment with overlapping chunks for better context continuity. Consider topic-segmented chunking using an LLM to identify discussion boundaries rather than relying purely on speaker turns.

**Multi-meeting intelligence:** Better cross-meeting queries - we track how topics and decisions evolve over time. Build a meeting timeline view. Detect contradictions between meetings ("In the Jan 15 standup they said launch was on track, but by Jan 20 they postponed it").

**UI:** Replace Streamlit with a proper React frontend. Add conversation memory so follow-up questions work naturally. Show source chunks inline with highlights.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/ingest` | POST | Upload and process a text transcript |
| `/api/ingest/audio` | POST | Upload an audio file, transcribe with speaker labels, and ingest |
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
│   ├── voice/           # Audio transcription (faster-whisper + pyannote)
│   └── api/             # FastAPI routes
├── ui/                  # Streamlit frontend
├── tests/               # pytest suite
├── data/                # Sample transcripts
└── docs/                # Architecture + ADRs
```
