from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from typing import Optional, Literal
from pathlib import Path
import os
import json

class RAGConfig(BaseModel):
    data_dir: Path = Path("data")
    persist_dir: Path = Path("vectorstore")
    chunk_size: int = 1500
    chunk_overlap: int = 200
    # Chunking mode: 'basic' uses RecursiveCharacterTextSplitter, 'structure' applies section-aware logic
    chunking_mode: Literal["basic", "structure"] = "basic"
    top_k: int = 10
    embedding_model: str = "models/text-embedding-004"
    llm_model: str = "gemini-2.5-flash"
    prompt_version: str = "v2"  # default uses präzisere Autor-Jahr Zitation
    # High-level retrieval mode orchestrating underlying strategy components
    # vector -> vectorstore only; keyword -> BM25 keyword index; hybrid -> fusion of both
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "vector"
    retrieval_strategy: Literal["similarity", "mmr"] = "similarity"
    # MMR tuning (only used when retrieval_strategy == "mmr")
    mmr_lambda_mult: float = 0.5  # balance relevance (1.0) vs diversity (0.0)
    mmr_fetch_k_factor: int = 4    # candidate multiplier relative to k
    mmr_min_fetch_k: int = 50      # floor for fetch_k to ensure diversity space
    # Reranking (Phase 2 scaffold)
    rerank_enable: bool = False
    rerank_model: str | None = None  # optional cross-encoder model id
    rerank_fetch_k_factor: int = 3   # future use for enlarged candidate pool
    rerank_fetch_k_max: int = 50      # upper bound for candidate pool size
    rerank_cache_max: int = 5000      # max cached (query, doc) scores
    rerank_overlap_weight: float = 1.0  # weight for unique token overlap component
    rerank_tfidf_weight: float = 0.0    # weight for simple TF-IDF relevance component (optional)
    # Adaptive k (Phase 2)
    adaptive_k: bool = False
    k_min: int = 5
    k_max: int = 15
    # Query expansion (Phase 2)
    query_expansion: bool = False
    query_expansion_max: int = 2  # max additional rewrites
    # Accurate token counting (optional Phase 2 improvement)
    accurate_token_count: bool = False  # if True and tiktoken available, use model tokenizer
    token_encoding_name: str | None = None  # override encoding (e.g., cl100k_base)
    # Answer generation mode
    answer_mode: Literal["text", "json"] = "text"

    # Pydantic v2: replace deprecated inner Config with model_config
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_env(cls) -> "RAGConfig":
        # Allow overrides via environment
        kwargs = {}
        for field in cls.model_fields:
            env_key = f"RAG_{field.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                if field in {"chunk_size", "chunk_overlap", "top_k", "mmr_fetch_k_factor", "mmr_min_fetch_k", "rerank_fetch_k_factor", "rerank_fetch_k_max", "rerank_cache_max", "k_min", "k_max", "query_expansion_max"}:
                    value = int(value)
                elif field in {"mmr_lambda_mult", "rerank_overlap_weight", "rerank_tfidf_weight"}:
                    value = float(value)
                elif field in {"rerank_enable", "adaptive_k", "query_expansion", "accurate_token_count"}:
                    value = value.lower() in {"1", "true", "yes", "on"}
                kwargs[field] = value
        return cls(**kwargs)

    def hash_relevant_params(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunking_mode": self.chunking_mode,
            "retrieval_mode": self.retrieval_mode,
            "embedding_model": self.embedding_model,
            "retrieval_strategy": self.retrieval_strategy,
            "top_k": self.top_k,
            "prompt_version": self.prompt_version,
            "mmr_lambda_mult": self.mmr_lambda_mult,
            "mmr_fetch_k_factor": self.mmr_fetch_k_factor,
            "mmr_min_fetch_k": self.mmr_min_fetch_k,
            "rerank_enable": self.rerank_enable,
            "rerank_model": self.rerank_model,
            "rerank_fetch_k_factor": self.rerank_fetch_k_factor,
            "rerank_fetch_k_max": self.rerank_fetch_k_max,
            "rerank_cache_max": self.rerank_cache_max,
            "adaptive_k": self.adaptive_k,
            "k_min": self.k_min,
            "k_max": self.k_max,
            "query_expansion": self.query_expansion,
            "query_expansion_max": self.query_expansion_max,
            "accurate_token_count": self.accurate_token_count,
            "token_encoding_name": self.token_encoding_name,
            "answer_mode": self.answer_mode,
            "rerank_overlap_weight": self.rerank_overlap_weight,
            "rerank_tfidf_weight": self.rerank_tfidf_weight,
        }

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2, default=str)
