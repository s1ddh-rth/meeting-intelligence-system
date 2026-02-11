# ADR-003: LLM Provider Selection

## Status
Accepted

## Context
The system requires an LLM for two tasks: (1) extracting structured data from transcripts at ingestion time, and (2) generating grounded answers to user queries. The solution must work for a demo/interview context with minimal cost, while demonstrating production-ready architecture.

## Options Considered
1. **OpenAI GPT-4o** - Industry standard, excellent quality, paid API
2. **Google Gemini 2.5 Flash** - Free tier available, good quality, fast
3. **Anthropic Claude Sonnet** - Strong reasoning, paid API
4. **Ollama (local)** - Fully offline, variable quality depending on model

## Decision
Multi-provider architecture with Gemini 2.5 Flash as primary, Claude Sonnet and Ollama as drop-in replacements.

## Rationale
- Gemini 2.5 Flash free tier: zero cost for development and demo (10 RPM)
- Provider abstraction (ABC + factory pattern) allows swapping providers via config
- Claude Sonnet available for higher-quality production use
- Ollama enables fully offline operation for environments without internet
- Provider-agnostic orchestration: chains call the same `generate()` interface regardless of backend
- Demonstrates vendor independence — a key production consideration

## Trade-offs Accepted
- Free tier rate limiting (10 RPM) requires retry logic with backoff
- Provider abstraction adds a layer of indirection
- Ollama quality depends on available local hardware and model selection
- Each provider has slightly different response metadata (token counts, etc.)

## Consequences
- Abstract `LLMProvider` base class defines the contract
- Factory function reads `LLM_PROVIDER` env var to instantiate the right provider
- All LLM calls go through the same async interface
- Rate limiting handled in the Gemini provider with exponential backoff
