from __future__ import annotations
"""Query expansion scaffold (Phase 2).

Provides lightweight, dependency-free heuristic expansions:
 - Synonym substitutions via a small curated map.
 - Decomposition: split multi-facet queries by ' and ', ' & ', or commas.

Future: integrate LLM rewrite when provider available; add multi-vector retrieval fusion.
"""
from typing import List, Set
import re

_SYNONYM_MAP = {
    "k8s": ["kubernetes"],
    "kubernetes": ["k8s"],
    "performance": ["throughput", "latency"],
    "scalability": ["scaling"],
    "scheduler": ["scheduling"],
}

_SPLIT_PATTERN = re.compile(r"\s+(and|&)\s+|, *")


def generate_expansions(query: str, max_new: int = 2) -> List[str]:
    """Return up to `max_new` expanded query variants (deterministic order).

    Strategy: union(synonym substitutions first, then simple decomposition pieces (>2 words)).
    Original query NOT included in the output list.
    """
    normalized = query.lower().strip()
    tokens = normalized.split()
    expansions: List[str] = []
    seen: Set[str] = set()
    # Synonym substitution (single token replacements)
    for i, tok in enumerate(tokens):
        if tok in _SYNONYM_MAP:
            for syn in _SYNONYM_MAP[tok]:
                variant = tokens.copy()
                variant[i] = syn
                candidate = " ".join(variant)
                if candidate != normalized and candidate not in seen:
                    seen.add(candidate)
                    expansions.append(candidate)
                    if len(expansions) >= max_new:
                        return expansions
    # Decomposition: split on connectors; keep fragments that have >=2 tokens
    parts = [p.strip() for p in _SPLIT_PATTERN.split(normalized) if p and p not in {"and", "&"}]
    for p in parts:
        if len(p.split()) >= 2 and p not in seen and p != normalized:
            seen.add(p)
            expansions.append(p)
            if len(expansions) >= max_new:
                break
    return expansions
