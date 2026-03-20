"""Tests for the vector store service."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exceptions import VectorStoreError
from vector_store import QueryResult, VectorStore


class TestVectorStore:
    """Tests for the VectorStore lazy singleton."""

    @patch("vector_store.chromadb")
    def test_add_and_query_roundtrip(self, mock_chromadb, monkeypatch):
        """Test that added chunks can be retrieved via query."""
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/test_chroma")

        # Set up mock collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.get.return_value = {"ids": []}
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["text one", "text two"]],
            "metadatas": [[{"source": "a.pdf", "page": 0}, {"source": "a.pdf", "page": 1}]],
            "distances": [[0.1, 0.5]],
        }

        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        from config import reset_config
        reset_config()

        store = VectorStore.get()

        # Add chunks
        embeddings = np.random.rand(2, 384).astype(np.float32)
        added = store.add_chunks(
            chunk_ids=["id1", "id2"],
            texts=["text one", "text two"],
            embeddings=embeddings,
            metadatas=[
                {"source": "a.pdf", "page": 0},
                {"source": "a.pdf", "page": 1},
            ],
        )
        assert added == 2

        # Query
        query_emb = np.random.rand(384).astype(np.float32)
        results = store.query(query_emb, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, QueryResult) for r in results)
        assert results[0].chunk_id == "id1"
        assert results[0].distance == 0.1

    @patch("vector_store.chromadb")
    def test_duplicate_handling(self, mock_chromadb, monkeypatch):
        """Test that duplicate chunks are skipped gracefully."""
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/test_chroma")

        mock_collection = MagicMock()
        # Simulate that "id1" already exists
        mock_collection.get.return_value = {"ids": ["id1"]}
        mock_collection.count.return_value = 1

        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        from config import reset_config
        reset_config()

        store = VectorStore.get()

        embeddings = np.random.rand(2, 384).astype(np.float32)
        added = store.add_chunks(
            chunk_ids=["id1", "id2"],
            texts=["text one", "text two"],
            embeddings=embeddings,
            metadatas=[{"source": "a.pdf"}, {"source": "a.pdf"}],
        )

        # Only id2 should be added (id1 is duplicate)
        assert added == 1
        # Verify add was called with only the new chunk
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        assert call_args.kwargs["ids"] == ["id2"]

    @patch("vector_store.chromadb")
    def test_empty_collection_query(self, mock_chromadb, monkeypatch):
        """Test querying an empty collection returns empty list."""
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/test_chroma")

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        from config import reset_config
        reset_config()

        store = VectorStore.get()
        query_emb = np.random.rand(384).astype(np.float32)
        results = store.query(query_emb, top_k=5)

        assert results == []

    def test_singleton_pattern(self):
        """Test that get() returns the same instance."""
        s1 = VectorStore.get()
        s2 = VectorStore.get()
        assert s1 is s2
