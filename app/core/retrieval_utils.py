from __future__ import annotations
from .config import RAGConfig

def build_search_kwargs(config: RAGConfig) -> dict:
    """Return search kwargs dict for the configured retrieval strategy.

    Centralizes logic for computing MMR fetch_k and lambda to avoid duplication.
    """
    if config.retrieval_strategy == "mmr":
        fetch_k = max(config.top_k * config.mmr_fetch_k_factor, config.mmr_min_fetch_k)
        return {"k": config.top_k, "fetch_k": fetch_k, "lambda_mult": config.mmr_lambda_mult}
    return {"k": config.top_k}
