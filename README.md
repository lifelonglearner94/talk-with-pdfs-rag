 # 🔬 talk-with-pdfs – Modular RAG System for Scientific Articles

Chat with collections of scientific PDF papers using a modular Retrieval-Augmented Generation (RAG) stack (LangChain + Chroma + Gemini) with a Streamlit UI, a FastAPI service scaffold and a CLI. The codebase is refactored to separate ingestion, indexing, retrieval, reranking and generation for easier extension, evaluation and observability.

## ✨ Key Features

| Area | Status |
|------|--------|
| Modular pipeline (`RAGPipeline`) | ✅ |
| Automatic index (re)build with param/file hashing | ✅ |
| Streamlit chat UI | ✅ |
| CLI (build / ask / list-sources / rebuild / reset) | ✅ |
| Citation handling: v1 (simple) / v2 (Autor‑Jahr) | ✅ (v2 default) |
| Config via env (`RAG_` prefix) | ✅ |
| Logging + timing decorator | ✅ (basic) |
| Tests (hashing, ingestion, retrieval kwargs, list sources) | ✅ |
| Retriever strategy switch (similarity / mmr + tunables) | ✅ |
| Reranking (candidate expansion, LRU cache, heuristics) | ✅ (scaffold + tunables) |
| Heuristic query expansion | ✅ (scaffold + logging) |
| Rich citation extraction (authors, year, pages) | 🚧 (in progress) |
| API (FastAPI endpoints: /ask, /sources, /health, /metrics, /eval) | 🚧 (~45% complete) |
| Structured JSON logging | ✅ (RAG_LOG_JSON=1) |

## 🧱 Architecture Overview

```
app/
   core/
      config.py         # RAGConfig (env + parameters, v2 prompt default)
      ingestion.py      # PDF loading (filters Zone.Identifier artifacts)
   splitting.py      # RecursiveCharacterTextSplitter wrapper (basic + structured splitter v0.3)
   embeddings.py     # Google embeddings + resilient backoff wrapper
   vectorstore.py    # Chroma persistence & source listing (vector backend abstraction)
      hashing.py        # Stable index hash (files + params)
      retriever.py      # Strategy-aware retriever factory
      retrieval_utils.py# Central search kwargs builder (MMR math)
      generator.py      # LLM chain assembly (prompt selection)
      prompting.py      # Versioned prompt templates (v1, v2)
      citations.py      # Citation key utilities (future enrichment)
      logging.py        # Logger + timing decorator
      models.py         # Dataclasses for answer & retrieval results
      rag_pipeline.py   # High-level orchestration
   interfaces/
      streamlit_app.py  # UI entrypoint
      cli.py            # CLI (build / rebuild / ask / list-sources / reset)
   tests/
      test_config.py
      test_hash_and_ingestion.py
      test_retrieval_kwargs.py
      test_list_sources.py
      test_ingestion_split_retrieve.py
run_app.sh            # Convenience launcher for UI
docker-start.sh       # Helper to run container with host PDFs
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
| `RAG_CHUNKING_MODE` | `basic` (char splitter) or `structure` (section-aware v0.1) | `basic` |
| `RAG_RERANK_ENABLE`  | Enable reranking layer (heuristic reordering) | `0` |
| `RAG_RERANK_FETCH_K_FACTOR` | Rerank candidate overfetch multiplier | `4` |
| `RAG_RERANK_FETCH_K_MAX` | Rerank fetch max candidates | `200` |
| `RAG_ADAPTIVE_K`     | Enable adaptive k selection for retrieval | `0` |
| `RAG_QUERY_EXPANSION`| Enable heuristic query expansion (synonyms, decomposition) | `0` |
| `RAG_QUERY_EXPANSION_MAX` | Max expansion variants | `2` |
| `RAG_VECTOR_BACKEND` | Vector backend to use (`chroma`/`faiss`/`milvus`) | `chroma` |
| `INDEX_FORMAT_VERSION` | Version marker for on-disk index format | `1` |
| `RAG_LOG_JSON`       | Emit structured JSON logs (one JSON per line) | `0` |
| `RAG_TOP_K` | Retrieval top-k | `10` |
| `RAG_EMBEDDING_MODEL` | Embedding model name | `models/text-embedding-004` |
| `RAG_LLM_MODEL` | LLM model name | `gemini-2.5-flash` |
| `RAG_PROMPT_VERSION` | Prompt template version (`v1` simple, `v2` Autor‑Jahr) | `v2` |
| `RAG_RETRIEVAL_STRATEGY` | `similarity` or `mmr` | `similarity` |
| `RAG_MMR_LAMBDA_MULT` | MMR relevance↔diversity balance (1=relevanz) | `0.5` |
| `RAG_MMR_FETCH_K_FACTOR` | Multiplier for candidate pool size | `4` |
| `RAG_MMR_MIN_FETCH_K` | Mindestanzahl Kandidaten (floor) | `50` |
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

### FastAPI service (experimental)

The repository contains an API scaffold that exposes async endpoints for programmatic access. Key endpoints (development):

- `GET /health` — basic health check
- `POST /ask` — async ask (returns request id / polling for results)
- `GET /sources` — list indexed source docs
- `POST /admin/init_pipeline` — admin: ensure/rebuild pipeline
- `GET /metrics` — Prometheus exposition (when enabled)

Use `./run_api.sh` to start the development API server (see script for env expectations).

### CLI
```bash
# Build index if needed
rag-pdf-chat build

# Force rebuild regardless of hash
rag-pdf-chat rebuild    # or: rag-pdf-chat build --force

# Ask a question
rag-pdf-chat ask "Was sind die wichtigsten Scheduling-Strategien in Kubernetes?"

# List currently indexed source document names
rag-pdf-chat list-sources

# Delete vectorstore (next query will rebuild from scratch)
rag-pdf-chat reset
```

If you change chunking mode, enable accurate token counting, or add PDFs, the index-hashing logic triggers a rebuild automatically; use `rebuild` to force proactively.

## 🧪 Testing & Tooling

```bash
uv run pytest -q          # run test suite
uv run ruff check .       # lint
uv run mypy app/core      # type check core modules
```
Tests include: env override config, hash invalidation, ingestion filtering of Zone.Identifier artifacts, retrieval kwargs (MMR boundaries), list_sources ordering & uniqueness, and (behind skips) pipeline answer flow.

The evaluation harness under `experiments/` now supports recall@k, and the Phase 2 additions add MRR/MAP deltas and rerank comparisons. Use the harness to compare retrieval strategies and track improvements.

### Lightweight Evaluation Harness (MVP)

An experimental retrieval/answer quality harness lives under `experiments/eval/`.

Contents (initial):
* `run_eval.py` – runs a set of gold questions against the current index.
* `gold_examples.jsonl` – each line: `{id, question, expected_keywords, source_papers}`.

It computes recall@k over expected `source_papers` and keyword coverage in the generated answer (if LLM enabled). Output JSON metrics are timestamped under `experiments/results/`.

Run (after placing PDFs & building index):
```bash
uv run python experiments/eval/run_eval.py --k 10
```
Future phases will extend with MRR / MAP and rerank comparisons.

## 🧩 Prompting / Versions
Prompts live in `prompting.py`.

| Version | Fokus | Zitation | Verwendung |
|---------|-------|----------|------------|
| v1 | Einfache grounded Antwort | (QuelleName) | Legacy / einfache Fälle |
| v2 (Default) | Wissenschaftlicher Ton, Autor‑Jahr Format | (Autor Jahr) pro Satz / kombiniert | Aktueller Standard |

Setze `RAG_PROMPT_VERSION=v1` für die frühere, kürzere Variante.

## 🪪 Citations & Metadata
Currently: filename → `citation_key` + `source_name`. Structured splitter adds `section` + `section_index` when `RAG_CHUNKING_MODE=structure`. Planned: author/year extraction (via PDF metadata or first-page heuristics) + page number propagation and grouping of chunks by source.

## 🗂️ Index Hashing
Hash includes: sorted list of PDF (name, mtime, size) + subset of config params + format version. Stored at `vectorstore/index_state.json`. Mismatch triggers rebuild.

## 🧠 Retrieval Strategies
Toggle similarity / mmr (Maximal Marginal Relevance) at runtime (sidebar or env). More strategies (hybrid keyword, rerank) will integrate via `RetrieverFactory` extension.

## 🪵 Logging & Timing
Logger (`rag` namespace) with millisecond timing decorator `@timed`. Set `RAG_LOG_LEVEL=DEBUG` for verbose traces. Enable structured JSON logs with `RAG_LOG_JSON=1` (one JSON object per line on stdout). Future: token accounting hooks.

## 🛣️ Roadmap (Planned Enhancements)
Short-term:
- Optional OpenTelemetry hooks
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
| Reranker appears slow | Reduce `RAG_RERANK_FETCH_K_MAX` or disable rerank for lower latency. |
| API returns no data | Ensure pipeline was initialized via `POST /admin/init_pipeline` or run initial `rag-pdf-chat build`. |

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

Version: 0.2.0 (siehe `pyproject.toml`). The enhancement plan (docs/ENHANCEMENT_PLAN.md) documents recent progress: Phase 1 & 2 are complete; Phase 3 (API + metrics + ingestion durability) is ~45% done. Falls ein altes `talk_with_pdfs.egg-info` Verzeichnis Version 0.1.0 zeigt, bitte `rm -rf talk_with_pdfs.egg-info && uv build` ausführen um zu aktualisieren.

---

Tip: Name PDFs like `Surname, 2024, Title of Paper.pdf` to future‑proof richer citation extraction in v3.

## Progress & Next Steps

- Phase 1 (chunking, metadata, hybrid retrieval, logging): ✅
- Phase 2 (reranking, adaptive k, expansion, JSON answer scaffolding): ✅
- Phase 3 (API, ingestion queue, vector backend abstraction, metrics): ~45% — remaining work: durable on-disk dedupe, analytics dashboard, nightly regression CI.

If you'd like, I can also:

- update `run_api.sh` docs and example `.env` snippets in this README
- add a minimal example curl sequence for `/ask` and `/eval/run`
- prepare a short CONTRIBUTING checklist for Phase 3 tasks


Pro-Tipp: Stelle spezifische Fragen: statt "Was steht in den PDFs?" lieber "Welche Scheduling-Optimierungen für Kubernetes werden im Paper von 2024 vorgeschlagen?".
