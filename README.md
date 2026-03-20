# PDF Chatbot — Offline RAG System

A production-grade, fully offline PDF chatbot that runs entirely on your local machine. Upload PDFs, ask questions, and get sourced answers — no external APIs, no cloud inference.

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

## Resource Requirements

| Resource | Requirement |
|---|---|
| RAM | 8 GB minimum (system total) |
| CPU | Any modern x86_64 (no GPU required) |
| Disk | ~2 GB for models + Ollama |
| OS | Windows, macOS, or Linux |

**This system is CPU-only.** No CUDA or GPU is assumed.

## Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose** (for containerized setup)
- **Ollama** — [Install Ollama](https://ollama.ai)

## Local Setup

```bash
# 1. Clone and enter project
cd pdf_chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env if needed

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

# 4. Open browser
# http://localhost:8501
```

## Ollama Model Setup

```bash
# Default model (small, fast)
ollama pull phi3

# Alternative models (edit OLLAMA_MODEL in .env)
ollama pull mistral
ollama pull llama3
ollama pull gemma:2b
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
2. Ask questions you know the answer to and note the confidence scores
3. Ask unrelated questions and note the confidence scores
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

## Limitations & Known Issues

- **First query is slow** — models load lazily on first use (~10–30 seconds)
- **OCR quality varies** — scanned PDFs depend on tesseract accuracy and scan quality
- **No table extraction** — complex tables in PDFs may not parse correctly
- **Single collection** — all PDFs share one vector store collection
- **No authentication** — the web UI is open to anyone on the network
- **Context window** — phi3 has a limited context; very long chunks may be truncated

## Swapping the LLM Model

1. Pull the new model: `ollama pull <model-name>`
2. Update `.env`: `OLLAMA_MODEL=<model-name>`
3. Restart the application

Popular alternatives: `mistral`, `llama3`, `gemma:2b`, `phi3:medium`
