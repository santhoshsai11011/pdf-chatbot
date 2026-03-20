# PDF Chatbot — Offline RAG System

A production-grade, fully offline PDF chatbot that runs entirely on your local machine. Upload PDFs, ask questions, and get sourced, confidence-scored answers — **no OpenAI, no Claude API, no cloud inference**.

Built with a strict 4-layer architecture, lazy-loaded models, memory-aware resource management, and a retrieval confidence guardrail that prevents hallucination when document context is insufficient.

## Architecture

```
┌─────────────────────────────────┐
│         UI Layer                │  app.py (Streamlit)
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│       Pipeline Layer            │  rag_pipeline.py, prompt_templates.py
│   (orchestration + guardrail)   │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│       Service Layer             │  embeddings.py, vector_store.py,
│   (stateless, single-purpose)   │  reranker.py, llm_interface.py,
│                                 │  pdf_loader.py, text_chunker.py
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   Infrastructure Layer          │  config.py, logger.py, exceptions.py
└─────────────────────────────────┘
```

**Strict dependency rule:** lower layers never import from upper layers. No service-layer file imports Streamlit. Only `rag_pipeline.py` orchestrates across services.

## Key Features

- **Fully offline** — runs 100% locally after initial model download
- **RAG pipeline** — PDF extraction → chunking → embedding → vector search → re-ranking → LLM generation
- **Confidence guardrail** — skips LLM generation when retrieval confidence is too low, preventing hallucination
- **Memory-aware** — monitors RAM via `psutil`, skips re-ranker when memory is low
- **Lazy loading** — all models load on first use, not at startup (< 2s startup time)
- **OCR fallback** — handles scanned PDFs via pytesseract
- **Streaming responses** — token-by-token output via Ollama
- **Embedding cache** — disk-cached embeddings keyed by content hash
- **Conversation history** — maintains context across questions

## Tech Stack

| Component | Technology |
|---|---|
| PDF Extraction | PyMuPDF + pytesseract (OCR fallback) |
| Chunking | Custom sliding window (500 words, 50 overlap) |
| Embeddings | `all-MiniLM-L6-v2` via Sentence Transformers |
| Vector Store | ChromaDB (persistent, L2 distance) |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Ollama (phi3 default, swappable) |
| UI | Streamlit |
| Config | python-dotenv + typed dataclass |
| Testing | pytest + pytest-mock |
| Deployment | Docker + Docker Compose |

## Resource Requirements

| Resource | Requirement |
|---|---|
| RAM | 8 GB minimum (system total) |
| CPU | Any modern x86_64 (no GPU required) |
| Disk | ~2 GB for models + Ollama |
| OS | Windows, macOS, or Linux |

**This system is CPU-only.** No CUDA or GPU is assumed.

## Project Structure

```
pdf_chatbot/
├── app.py                  # UI layer — Streamlit only
├── rag_pipeline.py         # Pipeline — orchestration + confidence guardrail
├── prompt_templates.py     # Pipeline — prompt construction
├── pdf_loader.py           # Service — PDF extraction + OCR
├── text_chunker.py         # Service — sliding window chunker
├── embeddings.py           # Service — lazy embedding model + cache
├── vector_store.py         # Service — ChromaDB wrapper
├── reranker.py             # Service — lazy cross-encoder
├── llm_interface.py        # Service — Ollama streaming client
├── config.py               # Infrastructure — typed config from .env
├── logger.py               # Infrastructure — logging setup
├── exceptions.py           # Infrastructure — custom exceptions
├── tests/
│   ├── conftest.py         # Shared fixtures
│   ├── test_pdf_loader.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_rag_pipeline.py
│   └── test_llm_interface.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .gitignore
```

## Prerequisites

- **Python 3.11** (3.14 does not have prebuilt wheels for ML packages)
- **Ollama** — [Install Ollama](https://ollama.ai)
- **Docker & Docker Compose** (optional, for containerized setup)

## Local Setup

```bash
# 1. Clone and enter project
git clone https://github.com/santhoshsai11011/pdf-chatbot.git
cd pdf-chatbot

# 2. Create virtual environment (use Python 3.11)
python3.11 -m venv venv
source venv/bin/activate        # Linux/macOS
source venv/Scripts/activate    # Windows (Git Bash)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env

# 5. Start Ollama and pull model
ollama serve &
ollama pull phi3

# 6. Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Docker Setup

```bash
# 1. Configure environment
cp .env.example .env

# 2. Build and start
docker-compose up --build -d

# 3. Pull the LLM model (first time only)
docker exec pdf-chatbot-ollama ollama pull phi3

# 4. Open http://localhost:8501
```

## How It Works

### RAG Pipeline Flow

```
User Question
    │
    ▼
1. Embed Query (all-MiniLM-L6-v2)
    │
    ▼
2. Vector Search (ChromaDB, top-10 candidates)
    │
    ▼
3. Confidence Check ──── score < threshold? ──→ Return uncertainty response
    │                                            (skip LLM entirely)
    ▼
4. Re-rank (cross-encoder, top-3)
    │     └── skipped if RAM < 2GB
    ▼
5. Build Prompt (system + few-shot + context + question)
    │
    ▼
6. Stream LLM Response (Ollama/phi3)
    │
    ▼
7. Return RAGResponse (answer + sources + confidence + metadata)
```

### Confidence Guardrail

The system computes a confidence score from retrieval distances:

```
confidence = 1.0 / (1.0 + mean(top_k_distances))
```

If confidence falls below the threshold (default 0.25), the LLM is **never called**. Instead, the user sees a yellow warning box explaining that relevant information wasn't found. This prevents hallucination and saves compute.

### Lazy Loading

No model loads at startup. Each heavy resource uses a singleton pattern that initializes on first use:

```
Startup (< 2 seconds) → no models in memory
First PDF upload       → embedding model loads (~90 MB)
First query            → ChromaDB + re-ranker (~160 MB) + Ollama generates
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `phi3` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_TEMPERATURE` | `0.1` | LLM temperature (0.0–1.0) |
| `OLLAMA_MAX_TOKENS` | `512` | Max tokens per response |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap words between chunks |
| `TOP_K_RETRIEVAL` | `10` | Candidates from vector search |
| `TOP_K_RERANK` | `3` | Results after re-ranking |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.25` | Min confidence to generate |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB storage path |
| `EMBEDDING_CACHE_DIR` | `./data/embed_cache` | Embedding cache path |
| `LOG_LEVEL` | `INFO` | Logging level |

## Running Tests

```bash
pytest tests/ -v
```

Tests use mocks for all ML models and external services — no Ollama or GPU needed.

## Confidence Threshold Tuning

The `RETRIEVAL_CONFIDENCE_THRESHOLD` controls when the system refuses to answer:

| Value | Behaviour |
|---|---|
| `0.1` | Very permissive — answers most questions, may hallucinate |
| `0.25` | **Default** — balanced between coverage and accuracy |
| `0.5` | Strict — only answers when very relevant content is found |
| `0.8` | Very strict — rarely answers, high precision |

**How to tune:**
1. Upload your target documents
2. Ask questions you know the answer to — note the confidence scores
3. Ask unrelated questions — note the confidence scores
4. Set the threshold between those two ranges

## Memory Usage Reference

| Component | Approximate RAM | Notes |
|---|---|---|
| Embedding model | ~90 MB | Loaded on first query/index |
| Cross-encoder re-ranker | ~110 MB | Loaded on first query; skipped if < 2 GB free |
| ChromaDB client | ~50 MB | Loaded on first store operation |
| Ollama (phi3) | ~2.5 GB | Separate process, managed by Ollama |
| Streamlit app | ~50 MB | Base application overhead |
| **Total peak** | **~2.8 GB** | Plus Ollama process |

## Swapping the LLM Model

```bash
# 1. Pull the new model
ollama pull mistral

# 2. Update .env
OLLAMA_MODEL=mistral

# 3. Restart the app
```

Popular alternatives: `mistral`, `llama3`, `gemma:2b`, `phi3:medium`

## Limitations & Known Issues

- **First query is slow** — models load lazily on first use (~10–30 seconds)
- **OCR quality varies** — scanned PDFs depend on tesseract accuracy and scan quality
- **No table extraction** — complex tables in PDFs may not parse correctly
- **Single collection** — all PDFs share one vector store collection
- **No authentication** — the web UI is open to anyone on the network
- **Context window** — phi3 has a limited context; very long chunks may be truncated
- **CPU inference** — LLM response generation is slower than GPU-accelerated setups
