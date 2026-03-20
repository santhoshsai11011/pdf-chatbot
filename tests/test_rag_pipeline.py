"""Tests for the RAG pipeline with confidence guardrail."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline import RAGResponse, _compute_confidence, process_query


class TestComputeConfidence:
    """Tests for the confidence scoring function."""

    def test_perfect_similarity(self):
        """Zero distance should give confidence of 1.0."""
        assert _compute_confidence([0.0, 0.0, 0.0]) == 1.0

    def test_moderate_distance(self):
        """Mean distance of 1.0 should give confidence of 0.5."""
        assert abs(_compute_confidence([1.0, 1.0]) - 0.5) < 1e-6

    def test_high_distance(self):
        """High distances should give low confidence."""
        score = _compute_confidence([10.0, 10.0])
        assert score < 0.1

    def test_empty_distances(self):
        """Empty distance list should give 0.0 confidence."""
        assert _compute_confidence([]) == 0.0


class TestGuardrail:
    """Tests for the retrieval confidence guardrail."""

    @patch("rag_pipeline.OllamaClient")
    @patch("rag_pipeline.Reranker")
    @patch("rag_pipeline.VectorStore")
    @patch("rag_pipeline.EmbeddingModel")
    def test_guardrail_triggers_low_confidence(
        self, mock_emb_cls, mock_vs_cls, mock_rr_cls, mock_llm_cls,
        monkeypatch,
    ):
        """Test that guardrail triggers when confidence < threshold."""
        monkeypatch.setenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.5")

        from config import reset_config
        reset_config()

        # Mock embedding model
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = np.random.rand(384).astype(
            np.float32
        )
        mock_emb_cls.get.return_value = mock_emb

        # Mock vector store — return high distances (low confidence)
        mock_result = MagicMock()
        mock_result.chunk_id = "id1"
        mock_result.text = "some text"
        mock_result.metadata = {"source": "test.pdf", "page": 0}
        mock_result.distance = 10.0  # Very high distance

        mock_store = MagicMock()
        mock_store.query.return_value = [mock_result]
        mock_vs_cls.get.return_value = mock_store

        # Mock LLM — should NOT be called
        mock_llm = MagicMock()
        mock_llm_cls.get.return_value = mock_llm

        response = process_query("irrelevant question")

        assert isinstance(response, RAGResponse)
        assert response.was_generated is False
        assert response.confidence < 0.5
        assert "could not find relevant information" in response.answer.lower()
        # LLM should not have been called
        mock_llm.stream_response.assert_not_called()

    @patch("rag_pipeline.OllamaClient")
    @patch("rag_pipeline.Reranker")
    @patch("rag_pipeline.VectorStore")
    @patch("rag_pipeline.EmbeddingModel")
    def test_guardrail_bypassed_high_confidence(
        self, mock_emb_cls, mock_vs_cls, mock_rr_cls, mock_llm_cls,
        monkeypatch,
    ):
        """Test that guardrail is bypassed when confidence >= threshold."""
        monkeypatch.setenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.25")

        from config import reset_config
        reset_config()

        # Mock embedding model
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = np.random.rand(384).astype(
            np.float32
        )
        mock_emb_cls.get.return_value = mock_emb

        # Mock vector store — return low distances (high confidence)
        mock_result = MagicMock()
        mock_result.chunk_id = "id1"
        mock_result.text = "Neural networks are fundamental to AI."
        mock_result.metadata = {"source": "test.pdf", "page": 0}
        mock_result.distance = 0.1  # Very low distance = high confidence

        mock_store = MagicMock()
        mock_store.query.return_value = [mock_result]
        mock_vs_cls.get.return_value = mock_store

        # Mock reranker
        from reranker import RankedResult

        mock_rr = MagicMock()
        mock_rr.rerank.return_value = [
            RankedResult(
                chunk_id="id1",
                text="Neural networks are fundamental to AI.",
                metadata={"source": "test.pdf", "page": 0},
                original_distance=0.1,
                rerank_score=0.9,
            )
        ]
        mock_rr_cls.get.return_value = mock_rr

        # Mock LLM — should be called
        mock_llm = MagicMock()
        mock_llm.stream_response.return_value = iter(
            ["The answer is ", "neural networks."]
        )
        mock_llm_cls.get.return_value = mock_llm

        response = process_query("What is AI?")

        assert isinstance(response, RAGResponse)
        assert response.was_generated is True
        assert response.confidence >= 0.25
        assert "neural networks" in response.answer.lower()
        # LLM should have been called
        mock_llm.stream_response.assert_called_once()

    @patch("rag_pipeline.OllamaClient")
    @patch("rag_pipeline.Reranker")
    @patch("rag_pipeline.VectorStore")
    @patch("rag_pipeline.EmbeddingModel")
    def test_was_generated_flag_correct(
        self, mock_emb_cls, mock_vs_cls, mock_rr_cls, mock_llm_cls,
        monkeypatch,
    ):
        """Test that was_generated flag is correctly set in both paths."""
        monkeypatch.setenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.5")

        from config import reset_config
        reset_config()

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = np.random.rand(384).astype(
            np.float32
        )
        mock_emb_cls.get.return_value = mock_emb

        # High distance = low confidence = guardrail triggers
        mock_result = MagicMock()
        mock_result.chunk_id = "id1"
        mock_result.text = "text"
        mock_result.metadata = {"source": "t.pdf", "page": 0}
        mock_result.distance = 100.0

        mock_store = MagicMock()
        mock_store.query.return_value = [mock_result]
        mock_vs_cls.get.return_value = mock_store

        response = process_query("anything")

        assert response.was_generated is False
        assert len(response.retrieval_scores) > 0
        assert response.retrieval_scores[0] == 100.0
