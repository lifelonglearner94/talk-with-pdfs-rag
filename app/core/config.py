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
    top_k: int = 10
    embedding_model: str = "models/text-embedding-004"
    llm_model: str = "gemini-2.5-flash"
    prompt_version: str = "v2"  # default uses präzisere Autor-Jahr Zitation
    retrieval_strategy: Literal["similarity", "mmr"] = "similarity"
    # MMR tuning (only used when retrieval_strategy == "mmr")
    mmr_lambda_mult: float = 0.5  # balance relevance (1.0) vs diversity (0.0)
    mmr_fetch_k_factor: int = 4    # candidate multiplier relative to k
    mmr_min_fetch_k: int = 50      # floor for fetch_k to ensure diversity space

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
                if field in {"chunk_size", "chunk_overlap", "top_k", "mmr_fetch_k_factor", "mmr_min_fetch_k"}:
                    value = int(value)
                elif field in {"mmr_lambda_mult"}:
                    value = float(value)
                kwargs[field] = value
        return cls(**kwargs)

    def hash_relevant_params(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "retrieval_strategy": self.retrieval_strategy,
            "top_k": self.top_k,
            "prompt_version": self.prompt_version,
            "mmr_lambda_mult": self.mmr_lambda_mult,
            "mmr_fetch_k_factor": self.mmr_fetch_k_factor,
            "mmr_min_fetch_k": self.mmr_min_fetch_k,
        }

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2, default=str)
