from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from .hashing import compute_index_hash, load_stored_hash, store_hash, load_index_state, INDEX_FORMAT_VERSION
from .config import RAGConfig
from .retrieval_utils import build_search_kwargs
from .logging import logger
import shutil
import json
from typing import Optional

class VectorStoreManager:
    def __init__(self, persist_dir: Path, embedding_fn):
        self.persist_dir = persist_dir
        self.embedding_fn = embedding_fn
        self.persist_dir.mkdir(exist_ok=True, parents=True)
        self.vectorstore: Chroma | None = None

    def needs_rebuild(self, docs: List[Document], config: RAGConfig) -> bool:
        pdf_paths = {Path(d.metadata.get('source', '')) for d in docs}
        pdf_paths = [p for p in pdf_paths if p and p.exists()]
        new_hash = compute_index_hash(pdf_paths, config)
        state = load_index_state(self.persist_dir)
        stored_hash = state.get("hash") if state else None
        stored_version = state.get("version") if state else None
        version_mismatch = stored_version is not None and stored_version != INDEX_FORMAT_VERSION
        if version_mismatch:
            logger.info("index.version.mismatch stored=%s current=%s -> rebuild", stored_version, INDEX_FORMAT_VERSION)
        changed = stored_hash != new_hash
        logger.debug("index.hash stored=%s new=%s changed=%s version_mismatch=%s", stored_hash, new_hash, changed, version_mismatch)
        return changed or version_mismatch

    def needs_rebuild_from_dir(self, data_dir: Path, config: RAGConfig) -> bool:
        """Compute index hash directly from filesystem PDFs without loading them.

        Avoids heavy document loading/splitting on startup when no rebuild is required.
        """
        pdf_paths = [p for p in Path(data_dir).glob("*.pdf") if p.exists() and ':Zone.Identifier' not in p.name]
        new_hash = compute_index_hash(pdf_paths, config)
        state = load_index_state(self.persist_dir)
        stored_hash = state.get("hash") if state else None
        stored_version = state.get("version") if state else None
        version_mismatch = stored_version is not None and stored_version != INDEX_FORMAT_VERSION
        if version_mismatch:
            logger.info("index.version.mismatch stored=%s current=%s -> rebuild", stored_version, INDEX_FORMAT_VERSION)
        changed = stored_hash != new_hash
        logger.debug("index.hash(fs) stored=%s new=%s changed=%s version_mismatch=%s", stored_hash, new_hash, changed, version_mismatch)
        return changed or version_mismatch

    def build(self, chunks: List[Document], config: RAGConfig):
        pdf_paths = {Path(d.metadata.get('source', '')) for d in chunks}
        pdf_paths = [p for p in pdf_paths if p and p.exists()]
        # Purge existing collection to avoid duplicates/stale data
        try:
            if self.persist_dir.exists():
                shutil.rmtree(self.persist_dir)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.warning("failed to purge persist_dir before rebuild; duplicates may remain", exc_info=True)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_fn,
            persist_directory=str(self.persist_dir)
        )
        # Persist a simple sources.json listing citation keys or source names
        try:
            sources = []
            for d in chunks:
                md = d.metadata or {}
                # prefer citation_key if present, otherwise source_name or filename
                s = md.get("citation_key") or md.get("source_name") or Path(md.get("source", "")).stem
                if s and s not in sources:
                    sources.append(s)
            # Write sources.json atomically: write to a temp file then replace
            try:
                tmp = self.persist_dir / "sources.json.tmp"
                tmp.write_text(json.dumps(sources, ensure_ascii=False), encoding="utf-8")
                tmp.replace(self.persist_dir / "sources.json")
            except Exception:
                # best-effort persistence; indexing should continue even if this fails
                logger.debug("could not persist sources.json (atomic write failed)")
        except Exception:
            # Outer guard: ensure build proceeds even if metadata extraction fails
            logger.debug("could not build sources list for persistence")
        new_hash = compute_index_hash(pdf_paths, config)
        store_hash(self.persist_dir, new_hash)

    def load(self):
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embedding_fn
        )

    def as_retriever(self, config: RAGConfig):
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
        search_kwargs = build_search_kwargs(config)
        logger.debug("retriever.build strategy=%s kwargs=%s", config.retrieval_strategy, search_kwargs)
        return self.vectorstore.as_retriever(
            search_type=config.retrieval_strategy,
            search_kwargs=search_kwargs,
        )

    # Convenience helpers for UI
    def list_sources(self) -> list[str]:
        """Return a sorted list of unique source file names currently in the vectorstore.
        If the vectorstore is not loaded yet, attempt to load it (if directory exists).
        """
        try:
            # Fast path: if we have a persisted sources.json, use it
            sj = self.persist_dir / "sources.json"
            if sj.exists():
                try:
                    data = json.loads(sj.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        return sorted({str(x) for x in data})
                except Exception:
                    pass
            if not self.vectorstore and any(self.persist_dir.glob("*")):
                self.load()
            if not self.vectorstore:
                return []
            # Chroma exposes get() to retrieve metadatas
            data = self.vectorstore.get(include=["metadatas"], where=None, limit=5_000)
            metas = data.get("metadatas", []) or []
            names = set()
            for md in metas:
                if not isinstance(md, dict):
                    continue
                # Prefer normalized source_name; otherwise derive from source path (stem only)
                src_name = md.get("source_name") or Path(md.get("source", "")).stem
                if src_name:
                    names.add(src_name)
            return sorted(names)
        except Exception:
            return []

    def reset(self):
        """Delete the persisted vectorstore completely (dangerous)."""
        if self.persist_dir.exists():
            shutil.rmtree(self.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore = None

    def create_ingestion_queue(self, index_name: Optional[str] = None, batch_size: int = 16, interval: float = 1.0):
        """Optional: Create an IngestionQueue wired to a vector backend.

        Note: This feature depends on optional modules not present by default in this repo.
        It is guarded and will raise a descriptive RuntimeError if the optional
        dependencies are missing. To enable, provide compatible implementations of
        `core.ingestion_queue` and `vector_backends`.
        """
        # Local dynamic import to keep optional dependency boundaries clear and silence static import errors
        try:
            import importlib
            ingestion_mod = importlib.import_module("core.ingestion_queue")
            backends_mod = importlib.import_module("vector_backends")
            worker_from_backend = getattr(ingestion_mod, "worker_from_backend")
            IngestionQueue = getattr(ingestion_mod, "IngestionQueue")
            ChromaBackend = getattr(backends_mod, "ChromaBackend")
        except Exception as e:  # pragma: no cover - defensive
            raise RuntimeError("Optional ingestion queue not available: missing core.ingestion_queue/vector_backends modules") from e

        backend = ChromaBackend(persist_directory=str(self.persist_dir))
        idx_name = index_name or "default"
        worker = worker_from_backend(backend, idx_name)
        return IngestionQueue(worker=worker, batch_size=batch_size, interval=interval)
