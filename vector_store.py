"""
Service Layer — ChromaDB Persistent Vector Store

Responsibility: Store and retrieve document chunk embeddings using
ChromaDB with persistent storage.

Permitted imports: Python stdlib, chromadb, numpy,
    infrastructure layer (config, logger, exceptions).
Must NOT import: streamlit, rag_pipeline, app, or any UI/pipeline module.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import os

# Disable ChromaDB telemetry to avoid noisy errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from config import get_config
from exceptions import VectorStoreError
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """A single vector search result.

    Attributes:
        chunk_id: The chunk's unique identifier.
        text: The chunk text content.
        metadata: Stored metadata (source, page, etc.).
        distance: L2 distance from query (lower = more similar).
    """

    chunk_id: str
    text: str
    metadata: Dict[str, object]
    distance: float


class VectorStore:
    """Lazy singleton wrapper around ChromaDB persistent client.

    The ChromaDB client is initialized on first use to keep startup fast.
    """

    _instance: Optional["VectorStore"] = None
    _client = None

    def __init__(self) -> None:
        """Private init — use VectorStore.get() instead."""

    @classmethod
    def get(cls) -> "VectorStore":
        """Return the singleton VectorStore, creating on first call.

        Returns:
            The singleton VectorStore instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None
        cls._client = None

    def _ensure_client(self) -> None:
        """Initialize the ChromaDB client if not yet created."""
        if self._client is None:
            import chromadb

            config = get_config()
            # ChromaDB persistent client — stores data in CHROMA_PERSIST_DIR
            logger.info(
                "[LAZY] Loading ChromaDB persistent client for the first "
                "time (persist_dir=%s)",
                config.chroma_persist_dir,
            )
            self._client = chromadb.PersistentClient(
                path=config.chroma_persist_dir
            )
            logger.info("ChromaDB client initialized")

    def _get_or_create_collection(
        self, name: str = "pdf_chunks"
    ) -> "chromadb.Collection":
        """Get or create a ChromaDB collection.

        Args:
            name: Collection name.

        Returns:
            A ChromaDB Collection object.
        """
        self._ensure_client()
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "l2"},
        )

    def add_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, object]],
        collection_name: str = "pdf_chunks",
    ) -> int:
        """Add chunks to the vector store, handling duplicates gracefully.

        Args:
            chunk_ids: Unique IDs for each chunk.
            texts: Text content of each chunk.
            embeddings: Embedding vectors as numpy array.
            metadatas: Metadata dicts for each chunk.
            collection_name: Target collection name.

        Returns:
            Number of chunks actually added (excluding duplicates).

        Raises:
            VectorStoreError: If the operation fails.
        """
        try:
            collection = self._get_or_create_collection(collection_name)

            # Check for existing IDs to handle duplicates
            existing = set()
            try:
                existing_results = collection.get(ids=chunk_ids)
                if existing_results and existing_results["ids"]:
                    existing = set(existing_results["ids"])
            except Exception:
                pass  # Collection might be empty

            # Filter out duplicates
            new_indices = [
                i for i, cid in enumerate(chunk_ids) if cid not in existing
            ]

            if not new_indices:
                logger.info(
                    "All %d chunks already exist — skipping insertion",
                    len(chunk_ids),
                )
                return 0

            new_ids = [chunk_ids[i] for i in new_indices]
            new_texts = [texts[i] for i in new_indices]
            new_embeddings = embeddings[new_indices].tolist()
            new_metadatas = [metadatas[i] for i in new_indices]

            # ChromaDB has a batch size limit; insert in batches
            batch_size = 500
            for start in range(0, len(new_ids), batch_size):
                end = start + batch_size
                collection.add(
                    ids=new_ids[start:end],
                    documents=new_texts[start:end],
                    embeddings=new_embeddings[start:end],
                    metadatas=new_metadatas[start:end],
                )

            added = len(new_ids)
            skipped = len(chunk_ids) - added
            logger.info(
                "Added %d chunks to collection '%s' (%d duplicates skipped)",
                added,
                collection_name,
                skipped,
            )
            return added

        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError("add_chunks", str(e)) from e

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        collection_name: str = "pdf_chunks",
    ) -> List[QueryResult]:
        """Query the vector store for the most similar chunks.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.
            collection_name: Collection to search.

        Returns:
            List of QueryResult sorted by distance (ascending).

        Raises:
            VectorStoreError: If the query fails.
        """
        try:
            collection = self._get_or_create_collection(collection_name)

            # Check if collection is empty
            if collection.count() == 0:
                logger.warning(
                    "Collection '%s' is empty — no results",
                    collection_name,
                )
                return []

            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            query_results: List[QueryResult] = []
            if results and results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    query_results.append(
                        QueryResult(
                            chunk_id=chunk_id,
                            text=results["documents"][0][i],
                            metadata=results["metadatas"][0][i],
                            distance=results["distances"][0][i],
                        )
                    )

            return query_results

        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError("query", str(e)) from e

    def delete_collection(
        self, collection_name: str = "pdf_chunks"
    ) -> None:
        """Delete a collection from the vector store.

        Args:
            collection_name: Name of the collection to delete.

        Raises:
            VectorStoreError: If deletion fails.
        """
        try:
            self._ensure_client()
            self._client.delete_collection(name=collection_name)
            logger.info("Deleted collection '%s'", collection_name)
        except Exception as e:
            raise VectorStoreError("delete_collection", str(e)) from e

    def list_collections(self) -> List[str]:
        """List all collection names in the vector store.

        Returns:
            List of collection name strings.
        """
        self._ensure_client()
        collections = self._client.list_collections()
        return [c.name for c in collections]

    def get_collection_count(
        self, collection_name: str = "pdf_chunks"
    ) -> int:
        """Get the number of items in a collection.

        Args:
            collection_name: Collection to count.

        Returns:
            Number of stored chunks.
        """
        try:
            collection = self._get_or_create_collection(collection_name)
            return collection.count()
        except Exception:
            return 0
