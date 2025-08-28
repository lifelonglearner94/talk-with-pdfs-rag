from __future__ import annotations
from .config import RAGConfig
import math

def choose_adaptive_k(similarities: list[float], config: RAGConfig) -> int:
    """Heuristic adaptive k selection based on similarity dispersion.

    Uses std deviation of similarity scores from an over-fetched candidate set.
    Higher dispersion => uncertainty => increase k. Low dispersion => lower k.
    Returns value within [k_min, k_max].
    """
    if not similarities:
        return config.top_k
    mean = sum(similarities) / len(similarities)
    var = sum((s - mean) ** 2 for s in similarities) / len(similarities)
    std = math.sqrt(var)
    # Piecewise heuristic thresholds (empirical defaults)
    if std < 0.005:
        return config.k_min
    if std < 0.015:
        return max(config.k_min, min(config.k_min + 1, config.top_k))
    if std < 0.045:
        return config.top_k
    if std < 0.06:
        return min(config.k_max, max(config.top_k + (config.k_max - config.top_k)//2, config.top_k))
    return config.k_max

def build_search_kwargs(config: RAGConfig) -> dict:
    """Return search kwargs dict for the configured retrieval strategy.

    Centralizes logic for computing MMR fetch_k and lambda to avoid duplication.
    """
    if config.retrieval_strategy == "mmr":
        fetch_k = max(config.top_k * config.mmr_fetch_k_factor, config.mmr_min_fetch_k)
        return {"k": config.top_k, "fetch_k": fetch_k, "lambda_mult": config.mmr_lambda_mult}
    return {"k": config.top_k}
