from app.core.retriever import KeywordRetriever, HybridRetriever
from langchain_core.documents import Document


def make_docs():
    texts = [
        ("Doc about Kubernetes orchestration and containers.", {"source_name": "k8s"}),
        ("Paper discussing fault tolerance in distributed systems.", {"source_name": "fault"}),
        ("Study on container scheduling algorithms for performance.", {"source_name": "sched"}),
    ]
    return [Document(page_content=t, metadata=md) for t, md in texts]


def test_keyword_retriever_basic():
    docs = make_docs()
    kr = KeywordRetriever(docs)
    kr.k = 2
    res = kr.invoke("scheduling container algorithms")
    assert res, "Expected at least one result"
    # Top document should be the one about scheduling algorithms or containers
    top_text = res[0].page_content.lower()
    assert "scheduling" in top_text or "container" in top_text


def test_hybrid_rrf_merges():
    # Build tiny fake vector retriever returning first two docs only
    docs = make_docs()
    class FakeVec:
        def invoke(self, q):
            return docs[:2]
    kr = KeywordRetriever(docs)
    kr.k = 3
    hybrid = HybridRetriever(FakeVec(), kr, k=3)
    out = hybrid.invoke("scheduling")
    # Should include scheduling doc even if vector retriever missed it (3rd doc)
    ids = {d.metadata['source_name'] for d in out}
    assert 'sched' in ids
