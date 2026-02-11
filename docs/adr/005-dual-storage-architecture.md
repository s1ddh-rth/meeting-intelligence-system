# ADR-005: Dual Storage Architecture

## Status
Accepted

## Context
The system handles two fundamentally different types of queries: (1) semantic questions requiring vector similarity search, and (2) structured questions about action items, decisions, and summaries that are better served by exact retrieval. Forcing all queries through vector search is suboptimal.

## Options Considered
1. **Vector-only** — Store everything in Qdrant, use vector search for all queries
2. **Relational-only** — Use PostgreSQL with full-text search, no embeddings
3. **Dual storage** — Qdrant for vectors + SQLite for pre-extracted structured data

## Decision
Dual storage: Qdrant for semantic vector search + SQLite for structured data.

## Rationale
- "List all action items" doesn't need vector search — it's a database query
- Pre-extracted structured data (via LLM at ingestion time) enables instant, accurate responses
- SQLite is zero-config and embedded — no additional Docker service needed
- Intent classification routes queries to the appropriate store
- Demonstrates understanding that RAG is not the answer to every query type

## Trade-offs Accepted
- Extraction at ingestion time adds latency and LLM cost to the ingestion pipeline
- Extracted data may miss nuance that vector search would capture
- Two data stores to maintain and keep in sync during re-ingestion
- SQLite lacks concurrent write support (acceptable for this use case)

## Consequences
- Ingestion pipeline includes an LLM extraction step after chunking and embedding
- Query chain classifies intent before choosing retrieval strategy
- STRUCTURED intent queries SQLite directly; SEMANTIC/SPEAKER use Qdrant
- Re-ingestion is idempotent: deletes existing data before inserting new
