import json, os, tempfile, shutil
from pathlib import Path
from langchain_core.documents import Document
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
        from app.core.reranker import SimpleReranker, RerankingRetriever
        docs = [Document(page_content="kubernetes performance scaling", metadata={"source":"dummy.pdf","section_index":0})]
        docs = self._enhance_metadata(docs)
        base = DummyRetriever(docs * 3)
        # no rerank layer needed here; use base retriever directly
        self._retriever = base

def test_query_log_includes_expansion_variants():
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            cfg = RAGConfig(query_expansion=True, query_expansion_max=2)
            pipe = DummyPipeline(cfg)
            res = pipe.answer("Kubernetes performance and scalability")
            assert res is not None
            log_path = Path("logs/query_log.jsonl")
            assert log_path.exists(), "query log not created"
            line = log_path.read_text().strip().splitlines()[-1]
            data = json.loads(line)
            assert data.get("query_expansion_used") is True
            variants = data.get("expansion_variants")
            assert isinstance(variants, list) and len(variants) > 0
            # Ensure original query not duplicated in variants
            assert all(v.lower() != "kubernetes performance and scalability".lower() for v in variants)
        finally:
            os.chdir(cwd)
