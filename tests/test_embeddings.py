"""Tests for the embedding service."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings import EmbeddingModel, _compute_batch_hash


class TestEmbeddingModel:
    """Tests for the EmbeddingModel lazy singleton."""

    @patch("embeddings.SentenceTransformer")
    def test_output_shape(self, mock_st_class):
        """Test that embeddings have the correct shape."""
        mock_model = MagicMock()
        # all-MiniLM-L6-v2 produces 384-dim embeddings
        mock_model.encode.return_value = np.random.rand(3, 384).astype(
            np.float32
        )
        mock_st_class.return_value = mock_model

        model = EmbeddingModel.get()
        texts = ["hello world", "test document", "another text"]
        embeddings = model.embed_documents(texts)

        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    @patch("embeddings.SentenceTransformer")
    def test_query_embedding_shape(self, mock_st_class):
        """Test that query embedding is a 1D vector."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384).astype(
            np.float32
        )
        mock_st_class.return_value = mock_model

        model = EmbeddingModel.get()
        embedding = model.embed_query("test query")

        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    @patch("embeddings.SentenceTransformer")
    def test_cache_hit_on_second_call(self, mock_st_class, tmp_path, monkeypatch):
        """Test that repeated calls use the disk cache."""
        monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(tmp_path))

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(2, 384).astype(
            np.float32
        )
        mock_st_class.return_value = mock_model

        from config import reset_config
        reset_config()

        model = EmbeddingModel.get()
        texts = ["hello", "world"]

        # First call — cache miss
        result1 = model.embed_documents(texts)
        assert mock_model.encode.call_count == 1

        # Second call — cache hit (encode should not be called again)
        result2 = model.embed_documents(texts)
        assert mock_model.encode.call_count == 1

        np.testing.assert_array_equal(result1, result2)

    @patch("embeddings.SentenceTransformer")
    def test_deterministic_embedding(self, mock_st_class):
        """Test that the same input produces the same output."""
        fixed_output = np.random.rand(1, 384).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fixed_output
        mock_st_class.return_value = mock_model

        model = EmbeddingModel.get()
        r1 = model.embed_query("test")
        r2 = model.embed_query("test")

        np.testing.assert_array_equal(r1, r2)

    def test_batch_hash_deterministic(self):
        """Test that batch hash is deterministic."""
        texts = ["hello", "world"]
        h1 = _compute_batch_hash(texts)
        h2 = _compute_batch_hash(texts)
        assert h1 == h2

    def test_batch_hash_different_inputs(self):
        """Test that different inputs produce different hashes."""
        h1 = _compute_batch_hash(["hello"])
        h2 = _compute_batch_hash(["world"])
        assert h1 != h2

    def test_singleton_pattern(self):
        """Test that get() returns the same instance."""
        m1 = EmbeddingModel.get()
        m2 = EmbeddingModel.get()
        assert m1 is m2
