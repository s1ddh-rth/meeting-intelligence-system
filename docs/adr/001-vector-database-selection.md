# ADR-001: Vector Database Selection

## Status
Accepted

## Context
The system requires a vector database to store and search meeting transcript embeddings. It needs to support metadata filtering (by speaker, meeting ID) for targeted retrieval, handle production-scale workloads, and run alongside the application in Docker.

## Options Considered
1. **ChromaDB** - Embedded Python vector store, easy setup, popular for prototyping
2. **pgvector** - PostgreSQL extension, combines relational + vector in one database
3. **Qdrant** - Standalone Rust-based vector database, Docker-native, rich filtering API

## Decision
Qdrant

## Rationale
- Production-grade architecture: Rust-based, memory-safe, high performance
- Superior metadata filtering via payload conditions — critical for speaker-aware queries
- Runs as a separate Docker service, demonstrating microservice separation
- REST and gRPC APIs with excellent Python client
- Supports cosine, dot product, and Euclidean distance metrics
- Actively maintained with strong community

## Trade-offs Accepted
- Requires a separate Docker service (vs embedded ChromaDB)
- Slightly more complex setup for local development
- Adds network dependency between app and Qdrant containers

## Consequences
- docker-compose includes a dedicated Qdrant service
- Vector store wrapper handles connection lifecycle and retry logic
- Metadata filtering enables speaker-specific and meeting-specific searches without re-embedding
