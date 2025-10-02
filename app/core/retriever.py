from __future__ import annotations
from .config import RAGConfig
from .retrieval_utils import build_search_kwargs
from langchain_core.documents import Document
from typing import List
import math

try:
    from rank_bm25 import BM25Okapi  # lightweight pure-python
    _HAS_BM25 = True
except Exception:  # pragma: no cover - optional dep
    _HAS_BM25 = False


class KeywordRetriever:
    """Simple BM25 keyword retriever over in-memory corpus.

    Built lazily from the underlying vectorstore documents (metadatas + content).
    Intended for small/medium corpora (<5k chunks) as Phase 1 scaffold.
    """
    def __init__(self, docs: List[Document]):
        self.docs = docs
        tokenized = [self._tokenize(d.page_content) for d in docs]
        if not _HAS_BM25:
            raise RuntimeError("rank_bm25 not installed; add to dependencies for keyword retrieval")
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re
        # Replace any non-word, non-space with space; collapse whitespace
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = cleaned.split()
        return [t for t in tokens if t.isalpha() or t.isalnum()]

    def invoke(self, query: str) -> List[Document]:  # LangChain retriever protocol subset
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        paired = list(enumerate(scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        k = getattr(self, 'k', 10)
        top = [self.docs[i] for i, s in paired[: k]]
        return top

    # duck type attributes
    k: int = 10

    def _aget_relevant_documents(self, *args, **kwargs):  # pragma: no cover - sync only
        raise NotImplementedError


class HybridRetriever:
    """Fuse vector + keyword retrievers via Reciprocal Rank Fusion (RRF)."""
    def __init__(self, vector_retriever, keyword_retriever: KeywordRetriever, k: int):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.k = k

    def invoke(self, query: str) -> List[Document]:
        # Try to enlarge candidate pools of both retrievers to at least k
        orig_vec_k = getattr(self.vector_retriever, 'k', None)
        orig_key_k = getattr(self.keyword_retriever, 'k', None)
        try:
            if orig_vec_k is not None and orig_vec_k < max(self.k, 2 * self.k):
                setattr(self.vector_retriever, 'k', max(self.k, 2 * self.k))
        except Exception:
            pass
        try:
            if orig_key_k is not None and orig_key_k < max(self.k, 2 * self.k):
                setattr(self.keyword_retriever, 'k', max(self.k, 2 * self.k))
        except Exception:
            pass
        vec_docs = self.vector_retriever.invoke(query)
        key_docs = self.keyword_retriever.invoke(query)
        # Restore original k values if changed
        if orig_vec_k is not None:
            try:
                setattr(self.vector_retriever, 'k', orig_vec_k)
            except Exception:
                pass
        if orig_key_k is not None:
            try:
                setattr(self.keyword_retriever, 'k', orig_key_k)
            except Exception:
                pass
        # Build rank maps
        def stable_key(doc: Document) -> str:
            md = doc.metadata or {}
            return md.get('chunk_id') or md.get('id') or md.get('_id') or md.get('source') or (hash(doc.page_content) and doc.page_content[:64])
        def rank_map(docs: List[Document]):
            return {stable_key(doc): r for r, doc in enumerate(docs, start=1)}
        rm_vec = rank_map(vec_docs)
        rm_key = rank_map(key_docs)
        all_docs = {stable_key(d): d for d in (vec_docs + key_docs)}
        fused = []
        for doc_id, doc in all_docs.items():
            r_vec = rm_vec.get(doc_id)
            r_key = rm_key.get(doc_id)
            score = 0.0
            if r_vec:
                score += 1.0 / (60 + r_vec)
            if r_key:
                score += 1.0 / (60 + r_key)
            fused.append((score, doc))
        fused.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in fused[: self.k]]

    def _aget_relevant_documents(self, *args, **kwargs):  # pragma: no cover - sync only
        raise NotImplementedError

class RetrieverFactory:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def _load_all_docs(self) -> List[Document]:
        # Chroma .get API: fetch all (limit large enough) including metadatas & documents
        raw = self.vectorstore.get(include=["documents", "metadatas"], limit=10_000)
        docs = raw.get("documents", []) or []
        metas = raw.get("metadatas", []) or []
        out: List[Document] = []
        for txt, md in zip(docs, metas):
            out.append(Document(page_content=txt, metadata=md or {}))
        return out

    def build(self, config: RAGConfig):
        search_kwargs = build_search_kwargs(config)
        vector_retriever = self.vectorstore.as_retriever(
            search_type=config.retrieval_strategy,
            search_kwargs=search_kwargs,
        )
        if config.retrieval_mode == "vector":
            return vector_retriever
        # Build keyword retriever
        docs = self._load_all_docs()
        keyword_ret = KeywordRetriever(docs)
        keyword_ret.k = config.top_k
        if config.retrieval_mode == "keyword":
            return keyword_ret
        # hybrid fusion
        return HybridRetriever(vector_retriever, keyword_ret, config.top_k)
