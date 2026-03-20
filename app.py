"""
UI Layer — Streamlit Application

Responsibility: Provide the web UI for the PDF chatbot. This is the ONLY
file permitted to import and use Streamlit.

Permitted imports: streamlit, pipeline layer, infrastructure layer.
This is the application entry point.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import streamlit as st

from config import get_config
from exceptions import OllamaConnectionError, PDFLoadError
from llm_interface import OllamaClient
from logger import get_logger
from pdf_loader import load_pdf
from rag_pipeline import RAGResponse, index_document, process_query_stream
from text_chunker import chunk_pages

logger = get_logger(__name__)


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📄",
        layout="wide",
    )

    _init_session_state()

    st.title("PDF Chatbot")
    st.caption("Ask questions about your uploaded PDF documents — fully offline")

    # Sidebar
    _render_sidebar()

    # Ollama health check on first load
    if not st.session_state.get("health_checked"):
        _check_ollama_health()
        st.session_state.health_checked = True

    # Show Ollama warning if down
    if not st.session_state.get("ollama_healthy", False):
        st.error(
            "⚠ Ollama is not reachable. Please start Ollama with "
            "`ollama serve` and refresh this page."
        )

    # Chat interface
    _render_chat()


def _init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[dict] = []
    if "indexed_docs" not in st.session_state:
        st.session_state.indexed_docs: List[str] = []
    if "conversation_pairs" not in st.session_state:
        st.session_state.conversation_pairs: List[Tuple[str, str]] = []


def _check_ollama_health() -> None:
    """Check Ollama connectivity and update session state."""
    try:
        client = OllamaClient.get()
        healthy = client.health_check()
        st.session_state.ollama_healthy = healthy
        if healthy:
            logger.info("Ollama health check passed")
        else:
            logger.warning("Ollama health check failed")
    except Exception as e:
        logger.warning("Ollama health check error: %s", e)
        st.session_state.ollama_healthy = False


def _render_sidebar() -> None:
    """Render the sidebar with PDF upload and document management."""
    with st.sidebar:
        st.header("Documents")

        # PDF upload
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.indexed_docs:
                    _process_uploaded_pdf(uploaded_file)

        # Indexed documents list
        if st.session_state.indexed_docs:
            st.subheader("Indexed Documents")
            for doc_name in st.session_state.indexed_docs:
                st.text(f"✓ {doc_name}")
        else:
            st.info("No documents indexed yet. Upload a PDF to get started.")

        # Clear button
        st.divider()
        if st.button("Clear All Data", type="secondary"):
            _clear_all_data()
            st.rerun()

        # Config info
        st.divider()
        config = get_config()
        st.caption(f"Model: {config.ollama_model}")
        st.caption(
            f"Confidence threshold: {config.retrieval_confidence_threshold}"
        )


def _process_uploaded_pdf(uploaded_file) -> None:
    """Process and index an uploaded PDF file.

    Args:
        uploaded_file: Streamlit UploadedFile object.
    """
    file_name = uploaded_file.name

    with st.sidebar:
        progress_bar = st.progress(0, text=f"Processing {file_name}...")

        try:
            # Step 1: Extract text
            progress_bar.progress(20, text=f"Extracting text from {file_name}...")
            pages = load_pdf(uploaded_file, source_name=file_name)

            # Step 2: Chunk
            progress_bar.progress(50, text=f"Chunking {file_name}...")
            chunks = chunk_pages(pages)

            # Step 3: Embed and index
            progress_bar.progress(70, text=f"Indexing {file_name}...")
            added = index_document(pages, chunks)

            # Done
            progress_bar.progress(100, text=f"Done! Added {added} chunks.")
            st.session_state.indexed_docs.append(file_name)
            logger.info(
                "Indexed '%s': %d pages, %d chunks, %d new",
                file_name,
                len(pages),
                len(chunks),
                added,
            )

        except PDFLoadError as e:
            progress_bar.empty()
            st.error(f"Failed to load {file_name}: {e.reason}")
            logger.error("PDF load error: %s", e)
        except Exception as e:
            progress_bar.empty()
            st.error(f"Error processing {file_name}: {e}")
            logger.error("Processing error for %s: %s", file_name, e)


def _clear_all_data() -> None:
    """Clear all indexed documents and chat history."""
    from vector_store import VectorStore

    try:
        store = VectorStore.get()
        store.delete_collection("pdf_chunks")
    except Exception as e:
        logger.warning("Error clearing vector store: %s", e)

    st.session_state.chat_history = []
    st.session_state.indexed_docs = []
    st.session_state.conversation_pairs = []
    logger.info("All data cleared")


def _render_chat() -> None:
    """Render the chat interface with message history and input."""
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                meta = message["metadata"]
                _render_response_metadata(meta)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.indexed_docs:
            st.warning("Please upload a PDF document first.")
            return

        # Add user message
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            _generate_response(prompt)


def _generate_response(question: str) -> None:
    """Generate and display a RAG response for the user's question.

    Args:
        question: The user's question text.
    """
    try:
        response_container = st.empty()
        full_answer = ""
        final_response: RAGResponse = None

        # Stream tokens from the pipeline
        for item in process_query_stream(
            question=question,
            history=st.session_state.conversation_pairs,
        ):
            if isinstance(item, RAGResponse):
                final_response = item
            elif isinstance(item, str):
                full_answer += item
                response_container.markdown(full_answer + "▌")

        # Finalize display
        if final_response is None:
            response_container.markdown(full_answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": full_answer}
            )
            return

        # Handle guardrail (low confidence) path
        if not final_response.was_generated:
            response_container.empty()
            st.warning(final_response.answer)
            st.metric(
                label="Retrieval Confidence",
                value=f"{final_response.confidence:.3f}",
            )
        else:
            response_container.markdown(final_response.answer)

        # Build metadata for storage
        metadata = {
            "confidence": final_response.confidence,
            "was_generated": final_response.was_generated,
            "sources": [
                {
                    "source": s.source,
                    "page": s.page,
                    "preview": s.text_preview,
                    "score": s.score,
                }
                for s in final_response.sources
            ],
        }

        # Show metadata
        _render_response_metadata(metadata)

        # Save to history
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": final_response.answer,
                "metadata": metadata,
            }
        )
        st.session_state.conversation_pairs.append(
            (question, final_response.answer)
        )

    except OllamaConnectionError as e:
        st.error(str(e))
        logger.error("Ollama connection error: %s", e)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error("Chat error: %s", e, exc_info=True)


def _render_response_metadata(metadata: dict) -> None:
    """Render confidence score and source references.

    Args:
        metadata: Dict with confidence, was_generated, and sources keys.
    """
    confidence = metadata.get("confidence", 0.0)
    was_generated = metadata.get("was_generated", True)

    if was_generated:
        st.caption(f"Confidence: {confidence:.3f}")

    sources = metadata.get("sources", [])
    if sources:
        with st.expander("📎 Sources", expanded=False):
            for i, src in enumerate(sources, 1):
                st.markdown(
                    f"**{i}. {src['source']}** (page {src['page']}, "
                    f"score: {src['score']:.3f})"
                )
                st.text(src.get("preview", "")[:200])
                st.divider()


if __name__ == "__main__":
    main()
