"""
Pipeline Layer — RAG Orchestration with Confidence Guardrail

Responsibility: Coordinate the full RAG pipeline: embed query, retrieve,
compute confidence, optionally re-rank, build prompt, and stream LLM response.
Implements the retrieval confidence guardrail.

Permitted imports: service layer modules, infrastructure layer modules.
Must NOT import: streamlit, app, or any UI module.
"""

from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import psutil

from config import get_config
from embeddings import EmbeddingModel
from llm_interface import OllamaClient
from logger import get_logger
from prompt_templates import build_prompt
from reranker import Reranker
from vector_store import VectorStore

logger = get_logger(__name__)


@dataclass
class SourceRef:
    """A reference to a source document chunk.

    Attributes:
        source: Original PDF filename.
        page: Page number.
        text_preview: First 200 characters of the chunk.
        score: Relevance score (rerank score or negative distance).
    """

    source: str
    page: int
    text_preview: str
    score: float


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline.

    Attributes:
        answer: The generated answer text.
        sources: List of source references.
        confidence: Retrieval confidence score (0.0–1.0).
        was_generated: True if LLM generated the answer; False if
            the guardrail triggered.
        retrieval_scores: Raw retrieval distances.
    """

    answer: str
    sources: List[SourceRef]
    confidence: float
    was_generated: bool
    retrieval_scores: List[float]


def _compute_confidence(distances: List[float]) -> float:
    """Compute a 0.0–1.0 confidence score from retrieval distances.

    Uses the formula: confidence = 1.0 / (1.0 + mean(distances))
    where lower L2 distance = higher confidence.

    Args:
        distances: List of L2 distances from vector search.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    if not distances:
        return 0.0
    mean_distance = sum(distances) / len(distances)
    return 1.0 / (1.0 + mean_distance)


def _log_memory(stage: str) -> None:
    """Log current memory usage at a pipeline stage.

    Args:
        stage: Human-readable name of the pipeline stage.
    """
    mem = psutil.virtual_memory()
    logger.info(
        "[MEMORY] %s — used: %.0f MB, available: %.0f MB (%.1f%%)",
        stage,
        mem.used / (1024 * 1024),
        mem.available / (1024 * 1024),
        mem.percent,
    )


def process_query(
    question: str,
    history: Optional[List[Tuple[str, str]]] = None,
    collection_name: str = "pdf_chunks",
) -> RAGResponse:
    """Run the full RAG pipeline for a user question (non-streaming).

    Flow: embed → search → confidence check → optional rerank → prompt → LLM.

    Args:
        question: The user's question.
        history: Optional conversation history as (question, answer) tuples.
        collection_name: ChromaDB collection to search.

    Returns:
        A RAGResponse with the answer, sources, and confidence info.
    """
    tokens = list(
        process_query_stream(question, history, collection_name)
    )

    # The last token is the RAGResponse object
    if tokens and isinstance(tokens[-1], RAGResponse):
        return tokens[-1]

    # Shouldn't happen, but handle gracefully
    answer = "".join(t for t in tokens if isinstance(t, str))
    return RAGResponse(
        answer=answer,
        sources=[],
        confidence=0.0,
        was_generated=True,
        retrieval_scores=[],
    )


def process_query_stream(
    question: str,
    history: Optional[List[Tuple[str, str]]] = None,
    collection_name: str = "pdf_chunks",
) -> Generator[object, None, None]:
    """Run the RAG pipeline with streaming LLM output.

    Yields string tokens during generation, then yields a final
    RAGResponse object with complete metadata.

    Args:
        question: The user's question.
        history: Optional conversation history.
        collection_name: ChromaDB collection to search.

    Yields:
        String tokens during generation, then a final RAGResponse.
    """
    config = get_config()
    _log_memory("Query start")

    # Step 1: Embed the query
    logger.info("Embedding query: '%s'", question[:80])
    embedding_model = EmbeddingModel.get()
    query_embedding = embedding_model.embed_query(question)
    _log_memory("After query embedding")

    # Step 2: Retrieve from vector store
    logger.info("Retrieving top-%d candidates", config.top_k_retrieval)
    store = VectorStore.get()
    results = store.query(
        query_embedding=query_embedding,
        top_k=config.top_k_retrieval,
        collection_name=collection_name,
    )
    _log_memory("After retrieval")

    # Step 3: Compute confidence
    distances = [r.distance for r in results]
    confidence = _compute_confidence(distances)
    logger.info(
        "Retrieval confidence: %.3f (threshold: %.3f, %d results)",
        confidence,
        config.retrieval_confidence_threshold,
        len(results),
    )

    # Step 4: Confidence guardrail
    if confidence < config.retrieval_confidence_threshold:
        logger.warning(
            "Low retrieval confidence: %.3f < threshold %.3f — "
            "skipping LLM generation",
            confidence,
            config.retrieval_confidence_threshold,
        )
        response = RAGResponse(
            answer=(
                "I could not find relevant information in the uploaded "
                "documents to answer this question."
            ),
            sources=[],
            confidence=confidence,
            was_generated=False,
            retrieval_scores=distances,
        )
        yield response.answer
        yield response
        return

    # Step 5: Re-rank (if RAM allows)
    candidates = [
        {
            "chunk_id": r.chunk_id,
            "text": r.text,
            "metadata": r.metadata,
            "distance": r.distance,
        }
        for r in results
    ]

    reranker = Reranker.get()
    ranked = reranker.rerank(
        query=question,
        candidates=candidates,
        top_k=config.top_k_rerank,
    )
    _log_memory("After re-ranking")

    # Step 6: Build context and prompt
    context_chunks = [
        {
            "text": r.text,
            "source": r.metadata.get("source", "unknown"),
            "page": r.metadata.get("page", "?"),
        }
        for r in ranked
    ]

    sources = [
        SourceRef(
            source=str(r.metadata.get("source", "unknown")),
            page=int(r.metadata.get("page", 0)),
            text_preview=r.text[:200],
            score=r.rerank_score,
        )
        for r in ranked
    ]

    prompt = build_prompt(context_chunks, question, history)
    logger.info("Prompt built (%d chars)", len(prompt))

    # Step 7: Stream LLM response
    _log_memory("Before LLM generation")
    llm = OllamaClient.get()
    answer_tokens: List[str] = []

    try:
        for token in llm.stream_response(prompt):
            answer_tokens.append(token)
            yield token
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        error_msg = f"Error generating response: {e}"
        yield error_msg
        answer_tokens = [error_msg]

    _log_memory("After LLM generation")

    # Step 8: Yield final RAGResponse with metadata
    full_answer = "".join(answer_tokens)
    response = RAGResponse(
        answer=full_answer,
        sources=sources,
        confidence=confidence,
        was_generated=True,
        retrieval_scores=distances,
    )
    yield response


def index_document(
    pages: list,
    chunks: list,
    collection_name: str = "pdf_chunks",
) -> int:
    """Embed and index document chunks into the vector store.

    Args:
        pages: List of PageContent (not used directly, for logging).
        chunks: List of Chunk objects to embed and store.
        collection_name: Target collection name.

    Returns:
        Number of new chunks added.
    """
    if not chunks:
        logger.warning("No chunks to index")
        return 0

    _log_memory("Indexing start")

    # Embed all chunks
    embedding_model = EmbeddingModel.get()
    texts = [c.text for c in chunks]
    embeddings = embedding_model.embed_documents(texts)
    _log_memory("After embedding documents")

    # Prepare metadata
    chunk_ids = [c.chunk_id for c in chunks]
    metadatas = [
        {
            "source": c.source,
            "page": c.page,
            "word_count": c.word_count,
        }
        for c in chunks
    ]

    # Store in vector DB
    store = VectorStore.get()
    added = store.add_chunks(
        chunk_ids=chunk_ids,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,
    )
    _log_memory("After vector store insertion")

    logger.info(
        "Indexed %d new chunks (total chunks: %d)",
        added,
        store.get_collection_count(collection_name),
    )
    return added
