# ADR-004: Chunking Strategy

## Status
Accepted

## Context
Meeting transcripts are conversational text with speaker turns, timestamps, and natural topic shifts. The chunking strategy directly impacts retrieval quality — the chunks become the atomic units that are embedded, searched, and returned as context to the LLM.

## Options Considered
1. **Fixed-size token chunks** — Split every N tokens regardless of content
2. **Topic-segmented chunks** — Use an LLM/model to detect topic boundaries
3. **Speaker-aware chunks** — Split on speaker changes, merge/split as needed

## Decision
Speaker-aware chunking with merge and split rules.

## Rationale
- Speaker turns are the natural unit of dialogue — preserving them maintains conversational context
- Metadata filtering by speaker enables "What did X say about..." queries
- Merging short consecutive same-speaker turns prevents tiny, low-value chunks
- Splitting long monologues at sentence boundaries prevents oversized chunks that dilute retrieval
- No LLM cost at chunking time (unlike topic segmentation)
- Deterministic and testable — no model inference involved in chunking

## Trade-offs Accepted
- Topic shifts within a speaker's turn may land in the same chunk
- Cross-speaker exchanges on one topic may be split across chunks
- Token estimation (word count * 1.3) is approximate, not exact
- Context overlap on split monologues adds slight redundancy

## Consequences
- Chunker produces rich metadata: meeting_id, speaker, timestamps, chunk_index, token count
- Qdrant payloads include speaker and meeting_id for filtered searches
- Retriever can combine metadata filtering with vector similarity
- Chunk boundaries align with conversation structure, improving citation quality
