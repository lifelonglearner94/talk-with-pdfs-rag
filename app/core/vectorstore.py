from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from .hashing import compute_index_hash, load_stored_hash, store_hash
from .config import RAGConfig
import shutil
import json

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
        stored = load_stored_hash(self.persist_dir)
        return stored != new_hash

    def build(self, chunks: List[Document], config: RAGConfig):
        pdf_paths = {Path(d.metadata.get('source', '')) for d in chunks}
        pdf_paths = [p for p in pdf_paths if p and p.exists()]
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_fn,
            persist_directory=str(self.persist_dir)
        )
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
        if config.retrieval_strategy == "mmr":
            fetch_k = max(config.top_k * config.mmr_fetch_k_factor, config.mmr_min_fetch_k)
            search_kwargs = {"k": config.top_k, "fetch_k": fetch_k, "lambda_mult": config.mmr_lambda_mult}
        else:
            search_kwargs = {"k": config.top_k}
        return self.vectorstore.as_retriever(
            search_type=config.retrieval_strategy,
            search_kwargs=search_kwargs
        )

    # Convenience helpers for UI
    def list_sources(self) -> list[str]:
        """Return a sorted list of unique source file names currently in the vectorstore.
        If the vectorstore is not loaded yet, attempt to load it (if directory exists).
        """
        try:
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
                src_name = md.get("source_name") or Path(md.get("source", "")).name
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
