from langchain_core.documents import Document
from app.core.reranker import SimpleReranker, RerankingRetriever


class BaseStubRetriever:
    """Returns documents in a fixed order that is intentionally suboptimal."""
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query: str):
        return list(self._docs)


def test_reranking_changes_order():
    query = "distributed scheduling performance"
    # Doc B should rank highest after rerank (overlap 3), then A (1), then C (0)
    doc_a = Document(page_content="performance and scaling considerations", metadata={"id": "A"})
    doc_b = Document(page_content="distributed scheduling performance study", metadata={"id": "B"})
    doc_c = Document(page_content="totally unrelated text", metadata={"id": "C"})
    base = BaseStubRetriever([doc_a, doc_b, doc_c])  # wrong initial order (A,B,C)
    rr = RerankingRetriever(base, SimpleReranker(), top_k=3)
    out = rr.invoke(query)
    ids = [d.metadata["id"] for d in out]
    assert ids[0] == "B", f"Expected B first after rerank, got {ids}"
    assert ids[-1] == "C"


def test_reranking_candidate_pool_and_cache():
    query = "alpha beta"
    # Construct docs where later docs have stronger overlap so fetch_k must exceed top_k to surface them.
    docs = [
        Document(page_content="alpha", metadata={"id": f"d{i}"}) for i in range(3)
    ] + [
        Document(page_content="alpha beta beta", metadata={"id": "target"})
    ]
    base = BaseStubRetriever(docs[:3])  # base returns only first 3 initially
    # Extend base to simulate larger corpus when k increased
    def invoke_with_dynamic_k(q):
        # if caller increased k beyond 3, return all docs
        k = getattr(base, 'k', 3)
        pool = docs if k > 3 else docs[:3]
        return pool[:k]
    base.invoke = invoke_with_dynamic_k  # type: ignore
    base.k = 3
    reranker = SimpleReranker(cache_max=10)
    rr = RerankingRetriever(base, reranker, top_k=3, fetch_k=10)
    out = rr.invoke(query)
    ids = [d.metadata['id'] for d in out]
    assert 'target' in ids, 'Expected candidate pool expansion to include target doc'
    # Second call should hit cache (cannot easily assert internal, but ensure deterministic order)
    out2 = rr.invoke(query)
    assert [d.metadata['id'] for d in out] == [d.metadata['id'] for d in out2]
    stats = reranker.last_stats()
    assert 'latency_sec' in stats and stats['latency_sec'] >= 0.0


def test_reranker_lru_cache_eviction():
    query = "alpha beta"
    # Prepare three docs; cache size 2 so one must be evicted after third unique insert.
    d1 = Document(page_content="alpha", metadata={"id": "d1"})
    d2 = Document(page_content="beta", metadata={"id": "d2"})
    d3 = Document(page_content="alpha beta", metadata={"id": "d3"})
    reranker = SimpleReranker(cache_max=2)
    # Access d1, then d2 -> cache: [d1,d2]
    reranker.score(query, [d1])
    reranker.score(query, [d2])
    # Re-access d1 to make it most recently used -> order becomes [d2,d1]
    reranker.score(query, [d1])
    # Insert d3 -> should evict d2 (least recently used)
    reranker.score(query, [d3])
    keys = reranker.cache_keys()
    # Extract doc texts in cache for clarity
    cached_texts = [k[1] for k in keys]
    assert d1.page_content in cached_texts, 'Expected d1 to remain (was recently used)'
    assert d3.page_content in cached_texts, 'Expected d3 to be cached'
    assert d2.page_content not in cached_texts, 'Expected d2 to be evicted as LRU'
