# ADR-006: Orchestration Approach

## Status
Accepted

## Context
The system needs to orchestrate multi-step pipelines: ingestion (parse → chunk → embed → store → extract) and query (classify → retrieve → prompt → generate). Orchestration frameworks like LangChain abstract this, but at the cost of complexity and debugging difficulty.

## Options Considered
1. **LangChain** — Popular framework, rich ecosystem, heavy abstraction layer
2. **LlamaIndex** — Data-focused framework, good for document pipelines
3. **Pydantic AI** — Lightweight agent framework from the Pydantic team
4. **Custom Python chain pattern** — Plain Python classes with explicit step orchestration

## Decision
Custom Python chain pattern with no external orchestration framework.

## Rationale
- Full control over execution flow — every step is visible and debuggable
- No framework lock-in or version compatibility issues
- Same retrieve-augment-generate pattern as frameworks, without the abstraction tax
- Each pipeline step is a separate, testable component with clear interfaces
- Dependency injection makes components swappable and mockable
- Demonstrates understanding of the underlying patterns, not just framework usage

## Trade-offs Accepted
- More boilerplate code than using LangChain/LlamaIndex
- No built-in tracing or observability (compensated with structlog)
- Must implement retry logic and error handling manually
- No access to framework ecosystem (prompt templates, memory, agents)

## Consequences
- IngestionChain and QueryChain are plain Python classes
- Components (parser, chunker, embedder, etc.) are injected at construction time
- Pipeline steps are explicit method calls with clear data flow
- Testing uses standard mocks — no framework-specific test utilities needed
- structlog provides observability at each pipeline step
