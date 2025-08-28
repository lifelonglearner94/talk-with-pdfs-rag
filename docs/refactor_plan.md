# Refactor Plan: RAG System for Scientific PDFs

## Objectives
1. Align codebase explicitly with Retrieval-Augmented Generation (RAG) architecture terminology.
2. Improve modularity, testability, and extensibility (e.g., swap vector DB, embedding model, LLM, loaders).
3. Separate concerns: ingestion, indexing, retrieval, generation, interface (CLI/UI), and configuration.
4. Strengthen metadata handling for scientific citations (authors, year, title, section, page numbers).
5. Introduce reproducible pipeline steps & caching (hashing, versioning of index build parameters).
6. Improve observability (logging, timings, token usage placeholders, quality metrics hooks).
7. Provide clear extension points (evaluation, reranking, summarization, multi-query expansion).
8. Ensure robust error handling & graceful degradation (e.g., missing API key, empty corpus, partial index).
9. Add minimal tests for ingestion, splitting, retrieval contract.
10. Prepare for multi-user session state (Streamlit) & potential backend API.

## Current Pain Points
- Monolithic `ScientificPDFSearcher` mixes ingestion, indexing, retrieval, prompting, and interaction.
- Lack of explicit domain model for pipeline stages (no `IngestionPipeline`, `IndexManager`, etc.).
- Metadata limited; no citation extraction (authors/year) -> reduces academic answer quality.
- Prompt defined inline; no prompt versioning / templating strategy.
- No configuration abstraction (hard-coded model names, chunk sizes, k=10, etc.).
- Streamlit app imports class directly; tight coupling; no service layer or API boundary.
- No evaluation / diagnostics (e.g., retrieval inspection, chunk provenance panel).
- No structured logging or metrics.
- Hash only considers filename + mtime; not parameters (chunk_size, overlap, model choices).
- No persistence versioning -> difficult to invalidate stale index when logic changes.

## Target Architecture
```
app/
  core/
    config.py              # Central settings (Pydantic / dataclass)
    models.py              # Domain data models (ChunkMetadata, RetrievalResult)
    ingestion.py           # PDF loading & document normalization
    splitting.py           # Text splitter factory & logic
    embeddings.py          # Embedding provider abstraction
    vectorstore.py         # Vector store adapter (Chroma + interface)
    retriever.py           # Retriever orchestration (k, strategies)
    prompting.py           # Prompt templates & builders
    generator.py           # LLM answer generation chain
    rag_pipeline.py        # High-level RAGPipeline (ingest -> retrieve -> generate)
    citations.py           # Metadata enrichment, citation formatting
    hashing.py             # Corpus + params hashing utilities
    logging.py             # Structured logging setup
  interfaces/
    cli.py                 # CLI entrypoint (replaces interactive loop in class)
    streamlit_app.py       # Refactored from start_app.py
    api.py (optional)      # FastAPI or similar (future)
  tests/
    test_ingestion.py
    test_splitter.py
    test_retrieval.py
main.py (removed; functionality migrated to modular pipeline + CLI/UI)
```

## Refactor Phases
### Phase 1: Extraction & Modularization
- Create `core/config.py` with RAGConfig (data_dir, persist_dir, chunk params, embedding model name, llm model name, top_k, prompt version).
- Move hashing logic to `hashing.py`; extend to include config parameters JSON.
- Split `ScientificPDFSearcher` into:
  - `PDFIngestor` (loader + raw docs)
  - `DocumentSplitter` (returns chunks)
  - `MetadataEnricher` (file name cleanup + placeholder for citation parsing)
  - `VectorStoreManager` (load/create, handles persistence + hash checking)
  - `RetrieverFactory` (produces retriever from store, supports similarity / mmr)
  - `AnswerGenerator` (wraps LLM + prompt)
  - `RAGPipeline` (compose steps for query: retrieve -> format -> generate -> attach sources)

### Phase 2: Prompt & Citation Enhancements
- Move prompt template into `prompting.py` with variables: context, question, citation_style, answer_language.
- Add citation formatting logic: APA-like (Author, Year) or fallback to filename.
- Attach page numbers and (future) section headings (if derivable) in context blocks.

### Phase 3: Configuration & Environment
- Use `.env` + `RAGConfig` (Pydantic BaseSettings) for override via env vars.
- Add config-driven switch for embedding / LLM models (allow easy future swap).
- Add `VECTORSTORE_PROVIDER` for potential Milvus / FAISS abstraction.

### Phase 4: Observability & Logging
- Introduce `structlog` or standard logging with JSON option.
- Log events: start_ingest, split_stats, build_index, load_index, query_received, retrieval_stats (k, latency), generation_latency.
- Add timing decorator utility.

### Phase 5: Testing & Quality Gates
- Implement unit tests: ingestion returns >0 docs; splitting yields consistent chunk_size boundaries; retrieval returns k docs; citation formatting robustness.
- Add a lightweight test PDF fixture with 2 short pages.
- Integrate `pytest` in `pyproject.toml` optional dev extras.

### Phase 6: Streamlit Refactor
- Rename `start_app.py` -> `interfaces/streamlit_app.py` (done).
- Inject `RAGPipeline` instead of directly calling searcher.
- Add sidebar controls: top_k slider, similarity vs MMR, show raw retrieved chunks toggle.
- Add expandable section to inspect retrieved chunks with metadata.

### Phase 7: CLI & Packaging
- Replace interactive loop with `interfaces/cli.py` using `argparse` to run queries or rebuild index.
- Expose entrypoint in `pyproject.toml` (e.g., `rag-pdf-chat`).

### Phase 8: Future Extensions (Scaffold, not full impl)
- Multi-query expansion (e.g., query rewriting to broaden recall) module placeholder.
- Reranking layer placeholder (e.g., cross-encoder) interface.
- Evaluation script for manual relevance checking.

## Detailed Component Contracts
### RAGConfig
- Inputs: env vars / defaults.
- Methods: `.hash_relevant_params()` -> dict subset for hashing.

### PDFIngestor
- Input: data_dir
- Output: List[Document]
- Errors: Directory missing, no PDFs.

### DocumentSplitter
- Input: List[Document], chunk_size, overlap
- Output: List[Document] (chunks with base metadata)

### MetadataEnricher
- Enhances: source_name, chunk_id, (future) authors, year, page.

### VectorStoreManager
- Methods: `load_or_build(docs, config)`; internal `_needs_rebuild()` with combined hash of files + params.
- Provides: `as_retriever(top_k, strategy)`.

### RAGPipeline
- Methods: `answer(question: str) -> AnswerResult` (contains text, sources, retrieved_chunks)

### AnswerGenerator
- Builds chain from prompt + llm; decoupled from retrieval logic.

## Hashing Strategy Improvement
- Current: filename + mtime.
- New: JSON of {files:[{name, mtime, size}], params: config.hash_relevant_params(), version: INDEX_FORMAT_VERSION} -> md5.
- Store in `vectorstore/index_state.json`.

## Metadata & Citation Plan
- Parse first page or filename for tentative title.
- (Optional future) Integrate lightweight PDF metadata extraction (e.g., `pymupdf` or `pypdf`) for authors/year.
- Add `citation_key` metadata (e.g., file_stem or derived BibKey).
- Source formatting: group retrieved chunks by `citation_key`; produce consolidated source list.

## Revised Prompt Template (illustrative)
```
System Role: You are a scientific research assistant.
You answer strictly from the provided document context.
If unsure or answer not present, say you cannot find it.

Context Chunks:
{context}

User Question:
{question}

Instructions:
- Cite sources inline using (Author Year) or (FileName) if unknown.
- Provide structured answer with sections if appropriate.
- List sources at the end under "Sources".
```

## Streamlit Enhancements
- Sidebar controls: chunk size (readonly if index already built), top_k, retrieval strategy.

---
### Cleanup (Aug 2025)
Residual legacy files `app/start_app.py` and `app/main.py` still present in repo were purged to avoid confusion. Current UI entrypoint: `app/interfaces/streamlit_app.py`. All pipeline access should go through `RAGPipeline`.
- Expander: Show retrieved chunk previews with metadata (score if available).
- Error panel if index outdated relative to new PDFs detected (button to rebuild).

## Migration Steps
1. Implement config + hashing modules.
2. Extract ingestion & splitting modules; adapt existing logic.
3. Implement vector store manager; migrate building/loading.
4. Create pipeline + answer generator using existing prompt.
5. Update Streamlit app to use pipeline.
6. Remove old monolithic class; keep thin wrapper for backward compatibility (deprecated).
7. Add tests & update README with new architecture diagram.

## Backward Compatibility
- Provide `ScientificPDFSearcher` shim that internally instantiates `RAGPipeline` to avoid breaking existing scripts temporarily.
- Deprecation warning on import.

## Risks & Mitigations
- Risk: Hash mismatch logic errors -> fallback to explicit rebuild command.
- Risk: Added abstraction overhead -> Keep simple dataclasses, avoid premature generalization.
- Risk: API key errors -> central validation in config.

## Success Criteria
- Running the Streamlit app yields identical or improved answers vs pre-refactor.
- Swapping `chunk_size` in config + rebuild triggers new index.
- Tests pass for ingestion, splitting, retrieval.
- Clear docs guiding extension (citations, reranking).

## Next Actions (Immediate)
- Create `core/` package with `__init__.py` and skeleton modules.
- Move prompt to `prompting.py`.
- Implement config + hashing; adapt existing main to use pipeline skeleton.

---
Prepared: 2025-08-24
