"""
Test fixtures shared across all test modules.

Provides sample PDF bytes, chunks, mock Ollama responses,
and configuration overrides.
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all lazy singletons before each test."""
    from embeddings import EmbeddingModel
    from vector_store import VectorStore
    from reranker import Reranker
    from llm_interface import OllamaClient
    from config import reset_config

    EmbeddingModel.reset()
    VectorStore.reset()
    Reranker.reset()
    OllamaClient.reset()
    reset_config()
    yield
    EmbeddingModel.reset()
    VectorStore.reset()
    Reranker.reset()
    OllamaClient.reset()
    reset_config()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate minimal valid PDF bytes for testing."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    text_point = fitz.Point(72, 72)
    page.insert_text(
        text_point,
        "This is a test document with enough words to pass the minimum "
        "chunk threshold. It contains information about artificial "
        "intelligence and machine learning. Neural networks are a key "
        "component of modern AI systems. Deep learning has revolutionized "
        "natural language processing and computer vision tasks.",
        fontsize=11,
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def sample_pdf_file(sample_pdf_bytes) -> io.BytesIO:
    """Provide a file-like object containing sample PDF bytes."""
    return io.BytesIO(sample_pdf_bytes)


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    """Generate a valid PDF with no text content."""
    import fitz

    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def sample_chunks():
    """Provide a list of sample Chunk objects."""
    from text_chunker import Chunk

    return [
        Chunk(
            chunk_id="abc123",
            text=(
                "This is a test document about artificial intelligence. "
                "Neural networks are a fundamental building block of "
                "modern AI systems. They learn patterns from data."
            ),
            source="test.pdf",
            page=0,
            word_count=28,
        ),
        Chunk(
            chunk_id="def456",
            text=(
                "Machine learning algorithms can be supervised or "
                "unsupervised. Supervised learning uses labeled data "
                "to train models for classification and regression tasks."
            ),
            source="test.pdf",
            page=0,
            word_count=24,
        ),
        Chunk(
            chunk_id="ghi789",
            text=(
                "Deep learning has revolutionized natural language "
                "processing. Transformer models like BERT and GPT "
                "have achieved state-of-the-art results on many NLP "
                "benchmarks and practical applications."
            ),
            source="test.pdf",
            page=1,
            word_count=28,
        ),
    ]


@pytest.fixture
def mock_ollama_response():
    """Provide a mock Ollama streaming response."""
    return [
        '{"response": "Based on", "done": false}',
        '{"response": " the document,", "done": false}',
        '{"response": " AI uses", "done": false}',
        '{"response": " neural networks.", "done": false}',
        '{"response": "", "done": true}',
    ]


@pytest.fixture
def mock_query_results():
    """Provide mock vector store query results."""
    from vector_store import QueryResult

    return [
        QueryResult(
            chunk_id="abc123",
            text="Neural networks are fundamental to AI.",
            metadata={"source": "test.pdf", "page": 0, "word_count": 7},
            distance=0.5,
        ),
        QueryResult(
            chunk_id="def456",
            text="Machine learning uses data patterns.",
            metadata={"source": "test.pdf", "page": 0, "word_count": 6},
            distance=0.8,
        ),
        QueryResult(
            chunk_id="ghi789",
            text="Deep learning revolutionized NLP.",
            metadata={"source": "test.pdf", "page": 1, "word_count": 5},
            distance=1.2,
        ),
    ]
