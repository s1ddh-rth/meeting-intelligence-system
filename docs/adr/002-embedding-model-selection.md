# ADR-002: Embedding Model Selection

## Status
Accepted

## Context
The system needs an embedding model to convert meeting transcript chunks into vectors for similarity search. The model must balance quality, speed, cost, and deployment simplicity. Embeddings are the foundation of retrieval quality.

## Options Considered
1. **OpenAI text-embedding-3-small** - High quality, cloud-hosted, costs money per token
2. **all-mpnet-base-v2** - Larger sentence-transformer (768-dim), best quality in SBERT family
3. **all-MiniLM-L6-v2** - Smaller sentence-transformer (384-dim), fast, popular for production

## Decision
all-MiniLM-L6-v2

## Rationale
- Zero cost: runs locally, no API key needed
- Fast: ~5x faster than all-mpnet-base-v2, critical for real-time ingestion
- 384-dimensional vectors: efficient storage in Qdrant, lower memory footprint
- Sufficient quality for meeting transcript text (short, conversational sentences)
- Well-tested and widely deployed in production systems
- Can be pre-loaded into Docker image at build time for fast startup

## Trade-offs Accepted
- Lower semantic quality than all-mpnet-base-v2 or OpenAI embeddings
- Fixed 384-dim vectors limit capacity for nuanced distinctions in very similar text
- English-optimised; may underperform on multilingual transcripts

## Consequences
- Embedder loads model once as a singleton (lazy-loaded on first use)
- Model is downloaded at Docker build time to avoid runtime delays
- 384-dim vectors configured in both the embedder and Qdrant collection
- Batch embedding used for ingestion efficiency
