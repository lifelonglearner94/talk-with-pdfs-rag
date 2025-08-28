# Project Cleanup & Refinement Plan

> Living document – always consult & update before making changes.

## 1. Goals
Make the codebase lean, consistent, test-backed, and extension‑ready while removing redundancy, dead/legacy artifacts, and sharpening dev ergonomics.

## 2. Current Inventory & Observations
- Core architecture already modular (`core/` separation good).
- Redundant retrieval kwargs logic duplicated in `vectorstore.as_retriever` and `RetrieverFactory.build`.
- Backup / legacy artifacts: `run_rag_experiments.py.backup`, `test_retriever_settings.py.backup`.
- Config env parsing manual (loop); could leverage `BaseSettings` or keep lightweight but add validation / `.env.example`.
- Logging: simple; no JSON / structured option.
- Tests minimal (only config env override + two pipeline tests, both heavily skipped without data/API key).
- Streamlit mutates `pipeline.config` directly; side-effects on retriever without encapsulation.
- Retrieval strategy params (MMR) adjusted in UI but no persistence across sessions (unless env overridden).
- Dockerfile: OK but could multi-stage for smaller image & include non-root user.
- No lint / formatting / type tooling (ruff, black, mypy, pre-commit).
- Experiment script (backup) valuable – should live under `experiments/` w/ README & maybe CLI flag for no LLM.
- Citation enrichment minimal; page numbers passed through but not surfaced richly.
- Duplication of capability: both `VectorStoreManager.as_retriever` & `RetrieverFactory.build` compute same search kwargs.
- Potential missing ignore rule for `.env`, evaluation outputs, experiment artifacts.

## 3. Redundancies / Candidates
| Area | Issue | Action |
|------|-------|--------|
| Retriever kwargs | Duplicated logic (MMR fetch_k, lambda) | Single helper function `build_search_kwargs(config)` |
| Backup scripts | *.backup clutter root | Relocate / rename or remove |
| Index rebuild logic | Hash computed twice conceptually (docs vs chunks) | Centralize hash input (use raw doc set only) |
| Direct config mutation in UI | Side-effectful | Provide `pipeline.update_retrieval_settings(**kwargs)` |
| Embeddings backoff | Wrapper already; ensure no unused imports (e.g., `Any` not imported) | Lint & prune |

## 4. Proposed Workstreams
### A. Repository Hygiene
- Add `.gitignore` covering caches, venv, vectorstore, local data, .env.
- Move backup scripts into `experiments/` (rename without `.backup`).
- Add `experiments/README.md` documenting usage.

### B. Code Refactors
- Consolidate retrieval kwargs logic.
- Add `pipeline.update_settings(top_k=?, strategy=?, mmr_*)` to encapsulate.
- Introduce helper for index rebuild decision (single source of truth, maybe in `vectorstore` still but factoring hash computation parameter).
- Optional: convert `RAGConfig.from_env` to inherit from `BaseSettings` (evaluate impact; keep current for simplicity initially).

### C. Testing Expansion
- Test: hash changes when (1) file size changes, (2) chunk_size changes.
- Test: MMR search kwargs (fetch_k >= top_k * factor, min floor).
- Test: ingestion skips `:Zone.Identifier` artifacts.
- Test: `vectorstore.list_sources()` returns unique sorted names.
- Provide lightweight synthetic PDF fixture (generated simple 2-page text) to avoid bundling heavy docs in CI.

### D. Developer Tooling
- Add `ruff` (format + lint), `mypy` (strict-ish), `pre-commit` config.
- Add `dev` extras: `pytest`, `ruff`, `mypy`, `pre-commit`.

### E. Documentation & Samples
- Add `.env.example` with commented defaults.
- Update `README.md` (remove outdated table rows / clarify prompt versions & MMR sliders).
- Add section on experiments script.
- Add architecture diagram (ASCII already; maybe refine or embed).

### F. CLI Enhancements
- Add subcommand: `list-sources`.
- Add subcommand: `rebuild` (force rebuild ignoring hash) and `reset`.

### G. Logging / Observability
- Add optional JSON logging via env `RAG_LOG_JSON=1`.
- Add debug log for index decision (old hash vs new hash).

### H. Docker Improvements (Optional Phase)
- Multi-stage build (builder w/ uv -> slim runtime) to reduce image size.
- Non-root user.
- Add healthcheck (`CMD curl -f http://localhost:8501/_stcore/health`) in docs.

### I. Experiment Harness
- Rename experiment script -> `experiments/run_rag_matrix.py`.
- Provide CLI args: `--no-llm`, `--limit N`, `--question <...>`.
- Output summary table (markdown) + JSON (already mostly there).

### J. Future (Defer unless time remains)
- Citation enrichment (author/year extraction heuristics).
- Reranker abstraction placeholder.
- FastAPI service interface.

## 5. Concrete Task Checklist (IDs)
Legend: [ ] Todo, [~] In Progress, [x] Done

### Hygiene
- [ ] T1: Add `.gitignore` (py cache, vectorstore, data, .env, experiments outputs)
- [ ] T2: Move backup scripts into `experiments/` (rename) & remove `.backup` suffix
- [ ] T3: Add `experiments/README.md`

### Refactors
- [ ] R1: Create `retriever_utils.py` or function inside `retriever.py` to build search kwargs
- [ ] R2: Use that function in both factory & vectorstore (remove duplication)
- [ ] R3: Add `RAGPipeline.update_settings()` encapsulating retriever rebuild
- [ ] R4: Add debug logging comparing stored vs new hash

### Tests
- [ ] TS1: Add fixture simple PDF generator (2 small PDFs) into `tests/fixtures`
- [ ] TS2: Test hash changes on config param change
- [ ] TS3: Test ingestion skip Zone.Identifier artifact
- [ ] TS4: Test MMR fetch_k computation boundaries
- [ ] TS5: Test `list_sources()` uniqueness & ordering

### Tooling
- [ ] D1: Add `ruff` + config (pyproject) & pre-commit
- [ ] D2: Add `mypy` basic config (optional strict for `app/core`)
- [ ] D3: Extend `[project.optional-dependencies].dev` with tooling
- [ ] D4: Document dev setup in README (lint, test)

### Docs
- [ ] DOC1: `.env.example`
- [ ] DOC2: Update README sections (Prompt versions, MMR explanation, experiments)
- [ ] DOC3: Add experiments usage docs

### CLI
- [ ] C1: Add `list-sources` command
- [ ] C2: Add `rebuild` command (force)
- [ ] C3: Add `reset` command (delete vectorstore)

### Logging
- [ ] L1: JSON logging option
- [ ] L2: More granular timing logs (index build steps)

### Docker (Optional Wave 2)
- [ ] K1: Multi-stage Dockerfile
- [ ] K2: Non-root user

### Experiments
- [ ] E1: Rename experiment script & parameterize
- [ ] E2: Summary markdown table generation improvement

### Deferred / Nice-to-have
- [ ] F1: Citation parsing heuristics (filename -> author/year extraction, structured)
- [ ] F2: Reranker abstraction placeholder
- [ ] F3: FastAPI interface

## 6. Sequencing / Batching
1. Hygiene + Refactor (T1–T3, R1–R4).
2. Tests + Tooling (TS*, D*).
3. CLI & Logging (C*, L*).
4. Docs & Examples (DOC*).
5. Experiments & Docker (E*, K*).
6. Deferred future tasks.

## 7. Acceptance Criteria
- All R* tasks reduce duplication (single retrieval kwargs builder).
- Test suite passes locally without external PDFs (self-contained fixture PDFs).
- Lint (ruff) & type check (mypy) produce no errors in core modules (warnings acceptable for external libs stubs).
- Running `rag-pdf-chat list-sources` works without requiring a question.
- Streamlit UI settings changes use pipeline update method (no direct config field mutation).
- README accurately reflects implemented capabilities.

## 8. Risk Notes & Mitigations
| Risk | Mitigation |
|------|------------|
| Increased complexity adding BaseSettings | Keep current env parser for now (document) |
| Slow tests due to embeddings | Use small fixture PDFs & mark live LLM tests as skipped unless key present |
| Docker size growth with tooling | Multi-stage build keeps runtime small |

## 9. Out of Scope (for now)
- Full metadata extraction from PDFs (needs OCR / NLP heuristics).
- Cross-encoder reranking model integration.
- Distributed / multi-tenant scaling concerns.

## 10. Initial Status Snapshot
All tasks currently open.

---
(Last updated: INITIAL COMMIT)
