from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from typing import Optional, Literal
from pathlib import Path
import os
import json

class RAGConfig(BaseModel):
    data_dir: Path = Path("data")
    persist_dir: Path = Path("vectorstore")
    # Optimal for scientific papers: larger chunks preserve context, moderate overlap ensures continuity
    chunk_size: int = 1200  # Sweet spot for academic content with citations
    chunk_overlap: int = 150  # ~12.5% overlap balances context without redundancy
    # Chunking mode: 'structure' is better for scientific papers (respects sections, tables, figures)
    chunking_mode: Literal["basic", "structure"] = "structure"
    # Higher top_k for scientific queries (complex questions benefit from more context)
    top_k: int = 12
    embedding_model: str = "models/text-embedding-004"
    llm_model: str = "gemini-2.5-flash"
    prompt_version: str = "v2"  # default uses präzisere Autor-Jahr Zitation
    # High-level retrieval mode orchestrating underlying strategy components
    # hybrid is best for scientific content: captures both semantic meaning and specific terminology
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    # MMR for scientific papers: reduces redundancy across similar sections/citations
    retrieval_strategy: Literal["similarity", "mmr"] = "mmr"
    # MMR tuning (only used when retrieval_strategy == "mmr")
    mmr_lambda_mult: float = 0.7  # Favor relevance but maintain some diversity for scientific breadth
    mmr_fetch_k_factor: int = 5    # Higher factor for better diversity pool
    mmr_min_fetch_k: int = 60      # Larger pool ensures diverse scientific perspectives
    # Reranking: Enable by default for scientific content (improves precision)
    rerank_enable: bool = True
    rerank_model: str | None = None  # optional cross-encoder model id
    rerank_fetch_k_factor: int = 4   # Fetch more candidates for better reranking
    rerank_fetch_k_max: int = 80      # Higher max for comprehensive scientific coverage
    rerank_cache_max: int = 5000      # max cached (query, doc) scores
    rerank_overlap_weight: float = 1.5  # Boost term overlap (important for scientific terminology)
    rerank_tfidf_weight: float = 1.0    # Enable TF-IDF for technical term matching
    # Adaptive k: Very useful for scientific queries (vary in complexity)
    adaptive_k: bool = True
    k_min: int = 6   # Minimum for basic factual queries
    k_max: int = 18  # Maximum for complex multi-faceted questions
    # Query expansion: Helpful for scientific terminology and synonyms
    query_expansion: bool = True
    query_expansion_max: int = 2  # Moderate expansion to avoid noise
    # Accurate token counting recommended for scientific content (precise chunk boundaries)
    accurate_token_count: bool = True  # Better handling of citations and special notation
    token_encoding_name: str | None = None  # override encoding (e.g., cl100k_base)
    # Answer generation mode
    answer_mode: Literal["text", "json"] = "text"
    # Preprocessing options
    remove_bibliography: bool = True  # Remove bibliography sections before indexing

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
                elif field in {"rerank_enable", "adaptive_k", "query_expansion", "accurate_token_count", "remove_bibliography"}:
                    value = value.lower() in {"1", "true", "yes", "on"}
                kwargs[field] = value
        return cls(**kwargs)

    def hash_relevant_params(self) -> dict:
        # Only include settings that affect chunk text/metadata or embeddings persisted in the index.
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunking_mode": self.chunking_mode,
            "embedding_model": self.embedding_model,
            # Token counting can alter chunk boundaries if used during split
            "accurate_token_count": self.accurate_token_count,
            "token_encoding_name": self.token_encoding_name,
            # Preprocessing affects the text that gets indexed
            "remove_bibliography": self.remove_bibliography,
        }

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2, default=str)
