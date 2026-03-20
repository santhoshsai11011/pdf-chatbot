"""
Infrastructure Layer — Typed Configuration

Responsibility: Load and validate all configuration from environment
variables and .env files into a typed dataclass singleton.
Permitted imports: Python stdlib + python-dotenv. No internal project imports.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import os


@dataclass(frozen=True)
class AppConfig:
    """Immutable, typed application configuration.

    All values are sourced from environment variables with sensible defaults.
    Use AppConfig.get() to obtain the singleton instance.
    """

    # LLM
    ollama_model: str = "phi3"
    ollama_base_url: str = "http://localhost:11434"
    ollama_temperature: float = 0.1
    ollama_max_tokens: int = 512

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k_retrieval: int = 10
    top_k_rerank: int = 3
    retrieval_confidence_threshold: float = 0.25

    # Storage
    chroma_persist_dir: str = "./data/chroma"
    embedding_cache_dir: str = "./data/embed_cache"

    # Logging
    log_level: str = "INFO"

    # Memory thresholds (MB)
    min_ram_for_reranker_mb: int = 2048
    low_ram_warning_mb: int = 1024


class _ConfigHolder:
    """Lazy singleton holder for AppConfig."""

    _instance: Optional[AppConfig] = None

    @classmethod
    def get(cls) -> AppConfig:
        """Return the singleton AppConfig, loading from env on first call.

        Returns:
            The application configuration instance.
        """
        if cls._instance is None:
            # Load .env file if present
            env_path = Path(".env")
            if env_path.exists():
                load_dotenv(env_path)

            cls._instance = AppConfig(
                ollama_model=os.getenv("OLLAMA_MODEL", "phi3"),
                ollama_base_url=os.getenv(
                    "OLLAMA_BASE_URL", "http://localhost:11434"
                ),
                ollama_temperature=float(
                    os.getenv("OLLAMA_TEMPERATURE", "0.1")
                ),
                ollama_max_tokens=int(
                    os.getenv("OLLAMA_MAX_TOKENS", "512")
                ),
                chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
                top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", "10")),
                top_k_rerank=int(os.getenv("TOP_K_RERANK", "3")),
                retrieval_confidence_threshold=float(
                    os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.25")
                ),
                chroma_persist_dir=os.getenv(
                    "CHROMA_PERSIST_DIR", "./data/chroma"
                ),
                embedding_cache_dir=os.getenv(
                    "EMBEDDING_CACHE_DIR", "./data/embed_cache"
                ),
                log_level=os.getenv("LOG_LEVEL", "INFO"),
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None


def get_config() -> AppConfig:
    """Public accessor for the application configuration.

    Returns:
        The singleton AppConfig instance.
    """
    return _ConfigHolder.get()


def reset_config() -> None:
    """Reset configuration singleton (for testing only)."""
    _ConfigHolder.reset()
