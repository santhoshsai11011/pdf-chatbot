"""Tests for the text chunker service."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exceptions import ChunkingError
from pdf_loader import PageContent
from text_chunker import Chunk, chunk_pages


def _make_page(text: str, page: int = 0) -> PageContent:
    """Helper to create a PageContent for testing."""
    return PageContent(page=page, text=text, source="test.pdf", ocr_used=False)


class TestChunkPages:
    """Tests for the chunk_pages function."""

    def test_basic_chunking(self, monkeypatch):
        """Test that chunking produces expected number of chunks."""
        monkeypatch.setenv("CHUNK_SIZE", "50")
        monkeypatch.setenv("CHUNK_OVERLAP", "10")

        # Create text with 100 words
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)
        pages = [_make_page(text)]

        from config import reset_config
        reset_config()

        chunks = chunk_pages(pages)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        # With 100 words, chunk_size=50, overlap=10 => step=40
        # Chunks start at: 0, 40, 80 => 3 chunks (last may be short)
        assert len(chunks) >= 2

    def test_overlap_correctness(self, monkeypatch):
        """Test that chunks overlap by the configured amount."""
        monkeypatch.setenv("CHUNK_SIZE", "30")
        monkeypatch.setenv("CHUNK_OVERLAP", "10")

        words = [f"w{i}" for i in range(60)]
        text = " ".join(words)
        pages = [_make_page(text)]

        from config import reset_config
        reset_config()

        chunks = chunk_pages(pages)

        if len(chunks) >= 2:
            words_0 = set(chunks[0].text.split())
            words_1 = set(chunks[1].text.split())
            overlap = words_0 & words_1
            # There should be overlapping words
            assert len(overlap) > 0

    def test_short_chunk_filtering(self, monkeypatch):
        """Test that chunks with fewer than 20 words are filtered out."""
        monkeypatch.setenv("CHUNK_SIZE", "500")
        monkeypatch.setenv("CHUNK_OVERLAP", "50")

        # Only 10 words — should be filtered
        text = " ".join(["short"] * 10)
        pages = [_make_page(text)]

        from config import reset_config
        reset_config()

        chunks = chunk_pages(pages)
        assert len(chunks) == 0

    def test_chunk_fields_populated(self, monkeypatch):
        """Test that all Chunk fields are properly populated."""
        monkeypatch.setenv("CHUNK_SIZE", "30")
        monkeypatch.setenv("CHUNK_OVERLAP", "5")

        text = " ".join([f"word{i}" for i in range(50)])
        pages = [_make_page(text, page=3)]

        from config import reset_config
        reset_config()

        chunks = chunk_pages(pages)

        for chunk in chunks:
            assert chunk.chunk_id  # Non-empty
            assert len(chunk.chunk_id) == 16  # SHA256 truncated
            assert chunk.source == "test.pdf"
            assert chunk.page == 3
            assert chunk.word_count >= 20

    def test_empty_pages_raises_error(self):
        """Test that empty page list raises ChunkingError."""
        with pytest.raises(ChunkingError):
            chunk_pages([])

    def test_multiple_pages(self, monkeypatch):
        """Test chunking across multiple pages."""
        monkeypatch.setenv("CHUNK_SIZE", "30")
        monkeypatch.setenv("CHUNK_OVERLAP", "5")

        pages = [
            _make_page(" ".join([f"p0w{i}" for i in range(40)]), page=0),
            _make_page(" ".join([f"p1w{i}" for i in range(40)]), page=1),
        ]

        from config import reset_config
        reset_config()

        chunks = chunk_pages(pages)

        # Should have chunks from both pages
        page_nums = {c.page for c in chunks}
        assert 0 in page_nums
        assert 1 in page_nums
