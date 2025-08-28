from __future__ import annotations
from .config import RAGConfig

class RetrieverFactory:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def build(self, config: RAGConfig):
        if config.retrieval_strategy == "mmr":
            fetch_k = max(config.top_k * config.mmr_fetch_k_factor, config.mmr_min_fetch_k)
            search_kwargs = {"k": config.top_k, "fetch_k": fetch_k, "lambda_mult": config.mmr_lambda_mult}
        else:
            search_kwargs = {"k": config.top_k}
        return self.vectorstore.as_retriever(
            search_type=config.retrieval_strategy,
            search_kwargs=search_kwargs
        )
