"""
Service Layer — Lazy-Loaded Cross-Encoder Re-Ranker

Responsibility: Re-rank retrieval candidates using a cross-encoder model
for improved relevance. Gracefully skips re-ranking when system RAM
is insufficient.

Permitted imports: Python stdlib, sentence_transformers, psutil,
    infrastructure layer (config, logger).
Must NOT import: streamlit, rag_pipeline, app, or any UI/pipeline module.
"""

from dataclasses import dataclass
from typing import List, Optional

import psutil

from config import get_config
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class RankedResult:
    """A re-ranked retrieval result.

    Attributes:
        chunk_id: The chunk's unique identifier.
        text: The chunk text content.
        metadata: Stored metadata dict.
        original_distance: Original vector distance from retrieval.
        rerank_score: Cross-encoder relevance score (higher = more relevant).
    """

    chunk_id: str
    text: str
    metadata: dict
    original_distance: float
    rerank_score: float


class Reranker:
    """Lazy singleton for the cross-encoder re-ranking model.

    The model is loaded on first use. If system RAM is too low,
    re-ranking is skipped and candidates are returned by original score.
    """

    _instance: Optional["Reranker"] = None
    _model = None
    _load_failed: bool = False

    def __init__(self) -> None:
        """Private init — use Reranker.get() instead."""

    @classmethod
    def get(cls) -> "Reranker":
        """Return the singleton Reranker instance.

        Returns:
            The singleton Reranker instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None
        cls._model = None
        cls._load_failed = False

    def _check_memory(self) -> bool:
        """Check if there's enough RAM to load the re-ranker.

        Returns:
            True if sufficient memory is available.
        """
        config = get_config()
        available_mb = psutil.virtual_memory().available / (1024 * 1024)

        if available_mb < config.min_ram_for_reranker_mb:
            logger.warning(
                "Insufficient RAM for re-ranker: %.0f MB available, "
                "need %d MB. Skipping re-ranking.",
                available_mb,
                config.min_ram_for_reranker_mb,
            )
            return False

        if available_mb < config.low_ram_warning_mb:
            logger.warning(
                "Low RAM warning: %.0f MB available (threshold: %d MB)",
                available_mb,
                config.low_ram_warning_mb,
            )

        return True

    def _ensure_model(self) -> bool:
        """Load the cross-encoder model if not yet loaded.

        Returns:
            True if model is available, False if loading was skipped.
        """
        if self._model is not None:
            return True

        if self._load_failed:
            return False

        if not self._check_memory():
            self._load_failed = True
            return False

        try:
            from sentence_transformers import CrossEncoder

            # ~110MB RAM — cross-encoder for passage re-ranking
            logger.info(
                "[LAZY] Loading re-ranker 'cross-encoder/ms-marco-MiniLM-L-6-v2' "
                "for the first time (~110MB RAM)"
            )
            mem_before = psutil.virtual_memory().used / (1024 * 1024)
            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            mem_after = psutil.virtual_memory().used / (1024 * 1024)
            logger.info(
                "Re-ranker loaded. RAM delta: %.0f MB "
                "(available: %.0f MB)",
                mem_after - mem_before,
                psutil.virtual_memory().available / (1024 * 1024),
            )
            return True

        except Exception as e:
            logger.warning("Failed to load re-ranker: %s", e)
            self._load_failed = True
            return False

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int = 3,
    ) -> List[RankedResult]:
        """Re-rank candidates using the cross-encoder model.

        If the model is unavailable (RAM constraint or load failure),
        falls back to returning the top_k candidates by original distance.

        Args:
            query: The user's query string.
            candidates: List of dicts with keys:
                chunk_id, text, metadata, distance.
            top_k: Number of top results to return.

        Returns:
            List of RankedResult sorted by relevance (best first).
        """
        if not candidates:
            return []

        model_available = self._ensure_model()

        if not model_available:
            logger.info(
                "Re-ranker unavailable — returning top-%d by original score",
                top_k,
            )
            return self._fallback_rank(candidates, top_k)

        # Check RAM before scoring
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        config = get_config()
        if available_mb < config.low_ram_warning_mb:
            logger.warning(
                "Low RAM (%.0f MB) — skipping re-ranking this query",
                available_mb,
            )
            return self._fallback_rank(candidates, top_k)

        # Build query-document pairs for cross-encoder scoring
        pairs = [(query, c["text"]) for c in candidates]

        try:
            scores = self._model.predict(pairs)
            scored = []
            for i, candidate in enumerate(candidates):
                scored.append(
                    RankedResult(
                        chunk_id=candidate["chunk_id"],
                        text=candidate["text"],
                        metadata=candidate["metadata"],
                        original_distance=candidate["distance"],
                        rerank_score=float(scores[i]),
                    )
                )

            # Sort by rerank_score descending (higher = more relevant)
            scored.sort(key=lambda r: r.rerank_score, reverse=True)
            return scored[:top_k]

        except Exception as e:
            logger.warning(
                "Re-ranking failed: %s — falling back to original scores", e
            )
            return self._fallback_rank(candidates, top_k)

    def _fallback_rank(
        self, candidates: List[dict], top_k: int
    ) -> List[RankedResult]:
        """Fallback: rank by original vector distance.

        Args:
            candidates: Candidate dicts with distance field.
            top_k: Number of results to return.

        Returns:
            List of RankedResult sorted by distance (ascending).
        """
        sorted_candidates = sorted(
            candidates, key=lambda c: c["distance"]
        )[:top_k]

        return [
            RankedResult(
                chunk_id=c["chunk_id"],
                text=c["text"],
                metadata=c["metadata"],
                original_distance=c["distance"],
                rerank_score=-c["distance"],  # Negative distance as pseudo-score
            )
            for c in sorted_candidates
        ]
