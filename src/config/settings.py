"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Meeting Intelligence System.

    All values can be overridden via environment variables or a .env file.
    """

    # LLM Provider
    llm_provider: str = "gemini"  # "gemini" | "anthropic" | "ollama"
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "meetings"

    # SQLite
    sqlite_path: str = "data/sqlite/meetings.db"

    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.2

    # Chunking
    max_chunk_tokens: int = 300
    min_chunk_tokens: int = 30

    # Voice transcription (optional)
    hf_token: str = ""
    whisper_model: str = "base"
    transcripts_dir: str = "data/transcripts"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
