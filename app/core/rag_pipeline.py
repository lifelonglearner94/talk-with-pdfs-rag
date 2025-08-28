from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from .config import RAGConfig
from .ingestion import PDFIngestor
from .splitting import DocumentSplitter
from .embeddings import EmbeddingProvider
from .vectorstore import VectorStoreManager
from .generator import AnswerGenerator
from .models import AnswerResult, ChunkMetadata, RetrievalResult
from .retriever import RetrieverFactory
from .retrieval_utils import build_search_kwargs
from .citations import derive_citation_key
from .logging import logger, timed

class RAGPipeline:
    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig.from_env()
        self.ingestor = PDFIngestor(self.config.data_dir)
        self.splitter = DocumentSplitter(self.config.chunk_size, self.config.chunk_overlap)
        self.embedding_provider = EmbeddingProvider(self.config.embedding_model)
        self.vs_manager = VectorStoreManager(self.config.persist_dir, self.embedding_provider.embeddings)
        self.answer_gen = AnswerGenerator(self.config)
        self._retriever = None
        self._retriever_factory: RetrieverFactory | None = None

    def _enhance_metadata(self, docs: List[Document]) -> List[Document]:
        enhanced = []
        for idx, d in enumerate(docs):
            source_file = d.metadata.get('source', '')
            source_name = source_file.split('/')[-1].split('.')[0]
            citation_key = derive_citation_key(source_file) if source_file else source_name
            d.metadata.update({
                'source_name': source_name,
                'chunk_id': f"{source_name}_chunk_{idx}",
                'citation_key': citation_key,
            })
            enhanced.append(d)
        return enhanced

    @timed("ensure_index")
    def ensure_index(self, force: bool = False):
        """Ensure an index is available.

        Parameters
        ----------
        force : bool
            If True, always rebuild the index even if hash unchanged.
        """
        logger.info("pipeline.ensure_index start data_dir=%s force=%s", self.config.data_dir, force)
        raw_docs = self.ingestor.ingest()
        chunks = self.splitter.split(raw_docs)
        chunks = self._enhance_metadata(chunks)
        # Decide rebuild
        if force or self.vs_manager.needs_rebuild(raw_docs, self.config):
            logger.info("index.rebuild %s", "forced" if force else "triggered")
            self.vs_manager.build(chunks, self.config)
        else:
            logger.info("index.load existing")
            self.vs_manager.load()
        self._retriever_factory = RetrieverFactory(self.vs_manager.vectorstore)
        self._retriever = self._retriever_factory.build(self.config)
        logger.info("pipeline.ensure_index done")

    @timed("answer")
    def answer(self, question: str) -> AnswerResult:
        logger.info("pipeline.answer question=%s", question)
        if not self._retriever:
            self.ensure_index()
        chain = self.answer_gen.build_chain(self._retriever)
        # Retrieve separately for metadata extraction
        retrieved = self._retriever.invoke(question)
        answer_text = chain.invoke(question)
        # Build source list
        sources_meta = []
        retrieval_results = []
        seen = set()
        for d in retrieved:
            md = ChunkMetadata(
                source_name=d.metadata.get('source_name', 'Unknown'),
                chunk_id=d.metadata.get('chunk_id', ''),
                page=d.metadata.get('page'),
            )
            retrieval_results.append(RetrievalResult(text=d.page_content, metadata=md))
            if md.source_name not in seen:
                sources_meta.append(md)
                seen.add(md.source_name)
        logger.info("pipeline.answer done sources=%d chunks=%d", len(sources_meta), len(retrieval_results))
        return AnswerResult(answer=answer_text, sources=sources_meta, raw_chunks=retrieval_results)

    def update_settings(self, **kwargs):
        """Update retrieval-related settings and rebuild the retriever in-place.

        Only affects in-memory retriever (does not trigger re-index). Safe for UI use.
        """
        mutable_fields = {
            "top_k",
            "retrieval_strategy",
            "mmr_lambda_mult",
            "mmr_fetch_k_factor",
            "mmr_min_fetch_k",
        }
        changed = False
        for k, v in kwargs.items():
            if k in mutable_fields and getattr(self.config, k) != v:
                setattr(self.config, k, v)
                changed = True
        if changed and self.vs_manager.vectorstore:
            if not self._retriever_factory:
                self._retriever_factory = RetrieverFactory(self.vs_manager.vectorstore)
            self._retriever = self._retriever_factory.build(self.config)
            logger.debug("pipeline.update_settings applied=%s new_kwargs=%s", changed, build_search_kwargs(self.config))
        return changed
