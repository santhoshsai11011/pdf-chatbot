"""
Service Layer — Lazy-Loaded Embedding Model

Responsibility: Provide document and query embedding via
SentenceTransformer('all-MiniLM-L6-v2') with disk caching.

Permitted imports: Python stdlib, sentence_transformers, joblib, psutil,
    infrastructure layer (config, logger, exceptions).
Must NOT import: streamlit, rag_pipeline, app, or any UI/pipeline module.
"""

import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import psutil

from config import get_config
from logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """Lazy singleton for the sentence-transformer embedding model.

    The model is loaded on first use and cached in memory.
    Embeddings are cached to disk keyed by content hash.
    """

    _instance: Optional["EmbeddingModel"] = None
    _model = None

    def __init__(self) -> None:
        """Private init — use EmbeddingModel.get() instead."""
        self._cache_dir: Optional[Path] = None

    @classmethod
    def get(cls) -> "EmbeddingModel":
        """Return the singleton EmbeddingModel, loading on first call.

        Returns:
            The singleton EmbeddingModel instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None
        cls._model = None

    def _ensure_model(self) -> None:
        """Load the model if not yet loaded."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # ~90MB RAM — lightweight embedding model suitable for 8GB systems
            logger.info(
                "[LAZY] Loading embedding model 'all-MiniLM-L6-v2' "
                "for the first time (~90MB RAM)"
            )
            mem_before = psutil.virtual_memory().used / (1024 * 1024)
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            mem_after = psutil.virtual_memory().used / (1024 * 1024)
            logger.info(
                "Embedding model loaded. RAM delta: %.0f MB "
                "(available: %.0f MB)",
                mem_after - mem_before,
                psutil.virtual_memory().available / (1024 * 1024),
            )

    def _get_cache_dir(self) -> Path:
        """Get or create the embedding cache directory."""
        if self._cache_dir is None:
            config = get_config()
            self._cache_dir = Path(config.embedding_cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed a list of document texts with disk caching.

        Args:
            texts: List of document strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        self._ensure_model()

        cache_dir = self._get_cache_dir()
        content_hash = _compute_batch_hash(texts)
        cache_path = cache_dir / f"docs_{content_hash}.npy"

        if cache_path.exists():
            logger.info("Embedding cache HIT for %d documents", len(texts))
            return np.load(cache_path)

        logger.info(
            "Embedding cache MISS — encoding %d documents", len(texts)
        )
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            batch_size=32,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Save to cache
        try:
            np.save(cache_path, embeddings)
        except OSError as e:
            logger.warning("Failed to write embedding cache: %s", e)

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            query: The query text to embed.

        Returns:
            numpy array of shape (embedding_dim,).
        """
        self._ensure_model()

        embedding = self._model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embedding[0], dtype=np.float32)


def _compute_batch_hash(texts: List[str]) -> str:
    """Compute a deterministic hash for a batch of texts.

    Args:
        texts: List of strings to hash.

    Returns:
        A hex digest string.
    """
    hasher = hashlib.sha256()
    for t in texts:
        hasher.update(t.encode("utf-8"))
    return hasher.hexdigest()[:20]
