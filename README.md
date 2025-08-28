 # 🔬 talk-with-pdfs – Modular RAG System for Scientific Articles

Chat with collections of scientific PDF papers using a modular Retrieval-Augmented Generation (RAG) stack (LangChain + Chroma + Gemini) with a Streamlit UI and a CLI. The codebase has been refactored to cleanly separate ingestion, indexing, retrieval, and generation for easier extension.

## ✨ Key Features

| Area | Status |
|------|--------|
| Modular pipeline (`RAGPipeline`) | ✅ |
| Automatic index (re)build with param/file hashing | ✅ |
| Streamlit chat UI | ✅ |
| CLI (build / ask) | ✅ |
| Basic citation keys (filename-based) | ✅ (authors/year TBD) |
| Config via env (`RAG_` prefix) | ✅ |
| Logging + timing decorator | ✅ (basic) |
| Tests (config + basic pipeline) | ✅ (more pending) |
| Retriever strategy switch (similarity / mmr) | ✅ |
| Future extension scaffolds (rerank, expansion) | ❌ (planned) |
| Rich citation extraction (authors, year, pages) | ❌ (planned) |
| Structured JSON logging | ❌ (planned) |

## 🧱 Architecture Overview

```
app/
   core/
      config.py        # RAGConfig (env + parameters)
      ingestion.py     # PDF loading
      splitting.py     # Chunking logic
      embeddings.py    # Embedding provider
      vectorstore.py   # Chroma persistence + hash management
      hashing.py       # File + params hash logic
      retriever.py     # RetrieverFactory (strategy aware)
      generator.py     # Answer generation chain
      prompting.py     # Prompt template versions
      citations.py     # Citation key utilities (placeholder for richer parsing)
      logging.py       # Logger + timing decorator
      models.py        # Dataclasses for results
      rag_pipeline.py  # High-level orchestration
   interfaces/
      streamlit_app.py # UI entrypoint
      cli.py           # CLI (build / ask)
   tests/
      test_config.py
      test_ingestion_split_retrieve.py
run_app.sh           # Convenience launcher for UI
```

Legacy monolithic code has been removed; all functionality now flows through the modular `RAGPipeline` and interface layers.

## ⚙️ Configuration

Environment variables (prefix `RAG_`) override defaults. Example:

| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_DATA_DIR` | Directory containing PDFs | `data` |
| `RAG_PERSIST_DIR` | Vector store directory | `vectorstore` |
| `RAG_CHUNK_SIZE` | Chunk size characters | `1500` |
| `RAG_CHUNK_OVERLAP` | Overlap characters | `200` |
| `RAG_TOP_K` | Retrieval top-k | `10` |
| `RAG_EMBEDDING_MODEL` | Embedding model name | `models/text-embedding-004` |
| `RAG_LLM_MODEL` | LLM model name | `gemini-2.5-flash` |
| `RAG_PROMPT_VERSION` | Prompt template version | `v1` |
| `RAG_RETRIEVAL_STRATEGY` | `similarity` or `mmr` | `similarity` |
| `RAG_LOG_LEVEL` | Log level (`DEBUG`, `INFO`, …) | `INFO` |

You must also set `GOOGLE_API_KEY` (handled by `langchain-google-genai`). Put all into a `.env` file (loaded by Streamlit or uv if configured):

```bash
echo "GOOGLE_API_KEY=your_key" >> .env
echo "RAG_TOP_K=8" >> .env
```

## 🛠 Installation

```bash
uv sync               # install dependencies
echo "GOOGLE_API_KEY=your_key" > .env
mkdir -p data
cp /path/to/papers/*.pdf data/
```

## 🚀 Usage

### Streamlit UI
```bash
./run_app.sh
# or
uv run streamlit run app/interfaces/streamlit_app.py
# or installed script
rag-pdf-chat-ui
```
Open http://localhost:8501.

### CLI
```bash
# Build / rebuild index explicitly
rag-pdf-chat build

# Ask a question
rag-pdf-chat ask "Was sind die wichtigsten Scheduling-Strategien in Kubernetes?"
```

If you change chunking or add PDFs, the hashing logic will trigger a rebuild automatically on next query; `build` forces it proactively.

## 🧪 Testing

Current tests (extend soon):
```bash
uv run pytest -q
```
Planned additions:
1. Hash invalidation test (change chunk_size -> new hash).
2. Retrieval contract (exact k when corpus large enough).
3. Chunk boundary invariants (no chunk > chunk_size + small tolerance).
4. Citation key presence.

## 🧩 Prompting
Prompts live in `prompting.py` (versioned). Version `v1` (German) enforces grounded answers and simple source citation. Future versions will introduce structured sectioned outputs and APA-like citations once metadata enrichment is implemented.

## 🪪 Citations & Metadata
Currently: filename → `citation_key` + `source_name`. Planned: author/year extraction (via PDF metadata or first-page heuristics) + page number propagation and grouping of chunks by source.

## � Index Hashing
Hash includes: sorted list of PDF (name, mtime, size) + subset of config params + format version. Stored at `vectorstore/index_state.json`. Mismatch triggers rebuild.

## 🧠 Retrieval Strategies
Toggle similarity / mmr (Maximal Marginal Relevance) at runtime (sidebar or env). More strategies (hybrid keyword, rerank) will integrate via `RetrieverFactory` extension.

## 🪵 Logging & Timing
Basic logger (`rag` namespace) with millisecond timing decorator `@timed`. Set `RAG_LOG_LEVEL=DEBUG` for verbose traces. Planned: JSON log mode & token accounting hooks.

## � Roadmap (Planned Enhancements)
Short-term:
- Structured JSON logging + optional OpenTelemetry hooks
- Rich citation enrichment (`citations.py` upgrades)
- Reranking interface (`reranking.py`) + query expansion (`query_expansion.py`)
- Additional tests & evaluation harness

Mid-term:
- API service (FastAPI) in `interfaces/api.py`
- Multi-user session management & conversation memory
- Configurable vector store backends (FAISS / Milvus abstraction)

Long-term:
- Automated evaluation & relevance labeling workflow
- Advanced summarization and multi-document synthesis modes

## 🧯 Troubleshooting
| Problem | Hint |
|---------|------|
| Missing API key | Ensure `GOOGLE_API_KEY` in `.env` or environment. |
| No PDFs indexed | Place `.pdf` files in `data/` (non-empty), restart or run `rag-pdf-chat build`. |
| Always rebuilding | Check system clock & file permissions; verify hash file `vectorstore/index_state.json`. |
| Slow first run | Embeddings computed only once; subsequent runs load persisted Chroma. |
| Retrieval returns few/no sources | Increase `RAG_TOP_K`, verify chunk size not too small. |

## 🤝 Contributing
1. Create feature branch.
2. Add / update tests for new behavior.
3. Run `pytest -q` before PR.
4. Update this README if user-facing changes.

## 📄 License
MIT (adjust if needed).

## 🙌 Acknowledgements
Built with LangChain, Chroma, Streamlit, and Google Gemini models.

---

Tip: Name PDFs like `Surname_Year_Title.pdf` to future‑proof citation extraction.

Pro-Tipp: Stelle spezifische Fragen: statt "Was steht in den PDFs?" lieber "Welche Scheduling-Optimierungen für Kubernetes werden im Paper von 2024 vorgeschlagen?".
