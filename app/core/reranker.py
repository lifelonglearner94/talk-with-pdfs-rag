from __future__ import annotations
"""Reranking component with heuristic + optional cross-encoder and stats.

Adds:
 - LRU cache for heuristic scores
 - Per-call stats (cache hit rate) exposed via last_stats()
 - Candidate pool size exposure in wrapper retriever (for query log enrichment)
"""
from typing import List, Optional, Dict, Tuple
import time
from collections import OrderedDict
from langchain_core.documents import Document

try:  # pragma: no cover - optional dependency
    from sentence_transformers import CrossEncoder  # type: ignore
    _HAS_CE = True
except Exception:  # pragma: no cover
    _HAS_CE = False


class SimpleReranker:
    def __init__(self, model_name: Optional[str] = None, cache_max: int = 0, overlap_weight: float = 1.0, tfidf_weight: float = 0.0):
        self.model_name = model_name
        self._model = None
        self._cache: "OrderedDict[Tuple[str, str], float]" = OrderedDict()
        self._cache_max = cache_max
        self._last_cache_hits: int = 0
        self._last_docs: int = 0
        self._last_latency: float = 0.0
        self.overlap_weight = overlap_weight
        self.tfidf_weight = tfidf_weight
        if model_name and _HAS_CE:
            try:  # pragma: no cover
                self._model = CrossEncoder(model_name)
            except Exception:  # pragma: no cover
                self._model = None

    def score(self, query: str, docs: List[Document]) -> List[float]:
        t0 = time.time()
        if self._model:  # pragma: no cover - external
            pairs = [(query, d.page_content) for d in docs]
            scores = list(self._model.predict(pairs))
            self._last_cache_hits = 0
            self._last_docs = len(docs)
            self._last_latency = time.time() - t0
            return scores
        q_tokens = {t for t in query.lower().split() if t}
        scores: List[float] = []
        cache_hits = 0
        for d in docs:
            dtoks = [t for t in d.page_content.lower().split() if t]
            if not dtoks:
                scores.append(0.0)
                continue
            cache_key = (query, d.page_content)
            if cache_key in self._cache:
                val = self._cache.pop(cache_key)
                self._cache[cache_key] = val
                scores.append(val)
                cache_hits += 1
                continue
            # Basic overlap metrics with normalization
            overlap = sum(1 for t in dtoks if t in q_tokens)
            unique_overlap = len({t for t in dtoks if t in q_tokens})
            qlen = max(1, len(q_tokens))
            dlen = max(1, len(dtoks))
            overlap_norm = (unique_overlap / qlen) + (overlap / dlen) * 0.01
            # Simple TF-IDF proxy: term frequency of query tokens normalized
            tfidf_proxy = sum(dtoks.count(t) for t in q_tokens) / dlen
            # Mild length penalty to avoid very long chunks dominating
            length_penalty = 1.0 / (1.0 + (dlen / 5000.0))
            sc = (self.overlap_weight * overlap_norm + self.tfidf_weight * tfidf_proxy) * length_penalty
            scores.append(sc)
            if self._cache_max:
                self._cache[cache_key] = sc
                if len(self._cache) > self._cache_max:
                    self._cache.popitem(last=False)
        self._last_cache_hits = cache_hits
        self._last_docs = len(docs)
        self._last_latency = time.time() - t0
        return scores

    def cache_keys(self) -> List[Tuple[str, str]]:  # pragma: no cover - trivial
        return list(self._cache.keys())

    def last_stats(self) -> Dict[str, float]:
        docs = self._last_docs or 1
        return {
            "cache_hits": self._last_cache_hits,
            "docs_scored": self._last_docs,
            "cache_hit_rate": self._last_cache_hits / docs,
            "latency_sec": self._last_latency,
            "overlap_weight": self.overlap_weight,
            "tfidf_weight": self.tfidf_weight,
        }


class RerankingRetriever:
    def __init__(self, base_retriever, reranker: SimpleReranker, top_k: int, fetch_k: Optional[int] = None):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.k = top_k
        self.fetch_k = fetch_k or top_k
        self._last_candidate_count: int | None = None

    def invoke(self, query: str) -> List[Document]:
        original_k = None
        if hasattr(self.base_retriever, 'k'):
            original_k = getattr(self.base_retriever, 'k')
            try:
                setattr(self.base_retriever, 'k', self.fetch_k)
            except Exception:  # pragma: no cover
                original_k = None
        base_docs = self.base_retriever.invoke(query)
        self._last_candidate_count = len(base_docs)
        if original_k is not None:
            try:  # pragma: no cover
                setattr(self.base_retriever, 'k', original_k)
            except Exception:  # pragma: no cover
                pass
        if len(base_docs) <= 1:
            return base_docs
        scores = self.reranker.score(query, base_docs)
        ranked = list(zip(scores, base_docs))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[: self.k]]

    def last_metadata(self) -> Dict[str, int | None]:  # pragma: no cover - simple
        return {"candidate_pool_size": self._last_candidate_count}

    def _aget_relevant_documents(self, *args, **kwargs):  # pragma: no cover - sync only
        raise NotImplementedError
