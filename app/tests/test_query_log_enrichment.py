import json, os, tempfile, shutil
from pathlib import Path
from langchain_core.documents import Document
from app.core.reranker import SimpleReranker, RerankingRetriever
from app.core.rag_pipeline import RAGPipeline
from app.core.config import RAGConfig

class DummyRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.k = len(docs)
    def invoke(self, query: str):
        return self.docs[:self.k]

class DummyPipeline(RAGPipeline):
    def ensure_index(self, force: bool = False):  # override to skip real ingestion
        docs = [Document(page_content="alpha beta gamma", metadata={"source":"dummy.pdf","section_index":0})]
        docs = self._enhance_metadata(docs)
        base = DummyRetriever(docs * 5)  # duplicate to create candidate pool
        reranker = SimpleReranker(cache_max=10)
        self._retriever = RerankingRetriever(base, reranker, top_k=1, fetch_k=5)


def test_query_log_includes_rerank_fields():
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            cfg = RAGConfig(rerank_enable=True)
            pipe = DummyPipeline(cfg)
            res = pipe.answer("alpha beta")
            assert res.answer is not None  # ensures pipeline ran
            log_path = Path("logs/query_log.jsonl")
            assert log_path.exists(), "query log not created"
            line = log_path.read_text().strip().splitlines()[-1]
            data = json.loads(line)
            # New fields
            assert "candidate_pool_size" in data and data["candidate_pool_size"] is not None
            assert "rerank_cache_hit_rate" in data
            # New explicit chosen_k field
            assert "chosen_k" in data and isinstance(data["chosen_k"], int)
        finally:
            os.chdir(cwd)
