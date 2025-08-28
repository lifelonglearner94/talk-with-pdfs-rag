import json, os, tempfile, shutil
from pathlib import Path
from langchain_core.documents import Document
from app.core.rag_pipeline import RAGPipeline
from app.core.config import RAGConfig
from app.core.reranker import SimpleReranker, RerankingRetriever

class DummyRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.k = len(docs)
    def invoke(self, q: str):
        return self.docs[: self.k]

class DummyPipeline(RAGPipeline):
    def ensure_index(self, force: bool = False):
        # Create docs with varying overlap so adaptive k logic has sims
        base_docs = [
            Document(page_content=f"kubernetes scheduling performance {i}", metadata={"source":f"doc{i}.pdf","section_index":0}) for i in range(12)
        ]
        enhanced = self._enhance_metadata(base_docs)
        base = DummyRetriever(enhanced)
        reranker = SimpleReranker(cache_max=10)
        self._retriever = RerankingRetriever(base, reranker, top_k=self.config.top_k, fetch_k=self.config.k_max)


def test_adaptive_k_with_rerank_and_expansion_logging():
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            cfg = RAGConfig(rerank_enable=True, adaptive_k=True, query_expansion=True, query_expansion_max=1, top_k=5, k_max=10)
            pipe = DummyPipeline(cfg)
            res = pipe.answer("Kubernetes performance and scheduling")
            assert res is not None
            log = Path("logs/query_log.jsonl").read_text().strip().splitlines()[-1]
            data = json.loads(log)
            # chosen_k should be within bounds and possibly > top_k due to adaptive logic
            assert cfg.k_min <= data["chosen_k"] <= cfg.k_max
            assert "expansion_recall_gain" in data
            # Ensure fields present for structured expansion tracking
            assert "expansion_added_sources" in data
        finally:
            os.chdir(cwd)
            shutil.rmtree(tmp)
