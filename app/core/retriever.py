from __future__ import annotations
from .config import RAGConfig
from .retrieval_utils import build_search_kwargs

class RetrieverFactory:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def build(self, config: RAGConfig):
        search_kwargs = build_search_kwargs(config)
        return self.vectorstore.as_retriever(
            search_type=config.retrieval_strategy,
            search_kwargs=search_kwargs,
        )
