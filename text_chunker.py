"""
Service Layer — Sliding Window Text Chunker

Responsibility: Split extracted page text into overlapping word-level
chunks suitable for embedding and retrieval.

Permitted imports: Python stdlib, infrastructure layer (config, logger, exceptions).
Must NOT import: streamlit, rag_pipeline, app, or any UI/pipeline module.
"""

from dataclasses import dataclass
from typing import List
import hashlib

from config import get_config
from exceptions import ChunkingError
from logger import get_logger
from pdf_loader import PageContent

logger = get_logger(__name__)

# Minimum number of words for a chunk to be kept
_MIN_CHUNK_WORDS = 20


@dataclass
class Chunk:
    """A text chunk ready for embedding.

    Attributes:
        chunk_id: Deterministic hash-based unique identifier.
        text: The chunk text content.
        source: Original PDF filename.
        page: Page number this chunk originated from.
        word_count: Number of words in this chunk.
    """

    chunk_id: str
    text: str
    source: str
    page: int
    word_count: int


def chunk_pages(pages: List[PageContent]) -> List[Chunk]:
    """Split page contents into overlapping word-level chunks.

    Uses a sliding window approach with configurable size and overlap.
    Chunks with fewer than 20 words are filtered out.

    Args:
        pages: List of PageContent from PDF extraction.

    Returns:
        List of Chunk objects ready for embedding.

    Raises:
        ChunkingError: If chunking fails unexpectedly.
    """
    if not pages:
        raise ChunkingError("No pages provided for chunking")

    config = get_config()
    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap

    if chunk_overlap >= chunk_size:
        raise ChunkingError(
            f"Overlap ({chunk_overlap}) must be less than "
            f"chunk size ({chunk_size})"
        )

    all_chunks: List[Chunk] = []

    for page_content in pages:
        try:
            page_chunks = _chunk_text(
                text=page_content.text,
                source=page_content.source,
                page=page_content.page,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_chunks.extend(page_chunks)
        except Exception as e:
            raise ChunkingError(
                f"Failed to chunk page {page_content.page} "
                f"of '{page_content.source}': {e}"
            ) from e

    logger.info(
        "Created %d chunks from %d pages (chunk_size=%d, overlap=%d)",
        len(all_chunks),
        len(pages),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks


def _chunk_text(
    text: str,
    source: str,
    page: int,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Chunk]:
    """Split a single page's text into overlapping chunks.

    Args:
        text: Raw text content.
        source: Source filename.
        page: Page number.
        chunk_size: Number of words per chunk.
        chunk_overlap: Number of overlapping words between chunks.

    Returns:
        List of Chunk objects from this page.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[Chunk] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        word_count = len(chunk_words)

        # Filter out short chunks
        if word_count >= _MIN_CHUNK_WORDS:
            chunk_text = " ".join(chunk_words)
            chunk_id = _generate_chunk_id(source, page, start)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    page=page,
                    word_count=word_count,
                )
            )

        start += step

        # Avoid infinite loop if step <= 0 (shouldn't happen with validation)
        if step <= 0:
            break

    return chunks


def _generate_chunk_id(source: str, page: int, start_word: int) -> str:
    """Generate a deterministic chunk ID based on source, page, and position.

    Args:
        source: Source filename.
        page: Page number.
        start_word: Starting word index.

    Returns:
        A short hex digest string.
    """
    raw = f"{source}::{page}::{start_word}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
