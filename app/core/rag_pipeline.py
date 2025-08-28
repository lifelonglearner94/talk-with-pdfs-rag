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
from .retrieval_utils import build_search_kwargs, choose_adaptive_k
from .citations import derive_citation_key
from .metadata_extraction import extract_basic_metadata
from .logging import logger, timed
from .reranker import SimpleReranker, RerankingRetriever
from .query_expansion import generate_expansions
import json, time
from pathlib import Path

class RAGPipeline:
    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig.from_env()
        self.ingestor = PDFIngestor(self.config.data_dir)
        self.splitter = DocumentSplitter(
            self.config.chunk_size,
            self.config.chunk_overlap,
            self.config,
        )
        self.embedding_provider = EmbeddingProvider(self.config.embedding_model)
        self.vs_manager = VectorStoreManager(
            self.config.persist_dir, self.embedding_provider.embeddings
        )
        self.answer_gen = AnswerGenerator(self.config)
        self._retriever = None
        self._retriever_factory: RetrieverFactory | None = None

    def _enhance_metadata(self, docs: List[Document]) -> List[Document]:
        enhanced = []
        # Track counts per (citation_key, section_index) for deterministic local ordering
        local_counters = {}
        for idx, d in enumerate(docs):
            source_file = d.metadata.get('source', '')
            source_name = source_file.split('/')[-1].split('.')[0]
            basic_meta = extract_basic_metadata(source_file) if source_file else {}
            # Normalize metadata values: Chroma requires metadata values to be
            # primitive types (str, int, float, bool, or None). Our extractor
            # or the ingestor may return lists/dicts (e.g. authors=['Carrion']).
            # Convert lists to comma-separated strings and dicts to JSON to
            # keep metadata human-readable and compatible with the vectorstore.
            def _normalize_value(v):
                # Allowed primitive types
                if v is None or isinstance(v, (str, int, float, bool)):
                    return v
                if isinstance(v, list):
                    # Convert list elements to strings and join
                    return ", ".join(str(x) for x in v)
                if isinstance(v, dict):
                    # Serialize dicts to compact JSON
                    try:
                        return json.dumps(v, ensure_ascii=False)
                    except Exception:
                        return str(v)
                # Fallback to string representation for other types
                return str(v)

            normalized_basic_meta = {k: _normalize_value(v) for k, v in (basic_meta or {}).items()}
            # Also normalize existing document metadata values in-place to avoid
            # passing complex types into the vectorstore.
            existing_meta = {k: _normalize_value(v) for k, v in (d.metadata or {}).items()}
            citation_key = derive_citation_key(source_file) if source_file else source_name
            sec_index = d.metadata.get('section_index', 0)
            counter_key = (citation_key, sec_index)
            local_counters.setdefault(counter_key, 0)
            local_idx = local_counters[counter_key]
            local_counters[counter_key] += 1
            # Start from existing metadata (normalized), then overlay our
            # computed/basic metadata so explicit extractor values take
            # precedence. This ensures all metadata values are primitives.
            merged_meta = {**existing_meta}
            merged_meta.update({
                'source_name': source_name,
                'chunk_id': f"{citation_key}:{sec_index}:{local_idx}",
                'citation_key': citation_key,
                'year': normalized_basic_meta.get('year'),
                'title': normalized_basic_meta.get('title'),
                'authors': normalized_basic_meta.get('authors'),
            })
            d.metadata = merged_meta
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
        base_ret = self._retriever_factory.build(self.config)
        # Optional reranking layer (Phase 2 scaffold)
        if getattr(self.config, 'rerank_enable', False):
            fetch_k = min(self.config.top_k * self.config.rerank_fetch_k_factor, self.config.rerank_fetch_k_max)
            reranker = SimpleReranker(self.config.rerank_model, cache_max=self.config.rerank_cache_max, overlap_weight=self.config.rerank_overlap_weight, tfidf_weight=self.config.rerank_tfidf_weight)
            self._retriever = RerankingRetriever(base_ret, reranker, self.config.top_k, fetch_k=fetch_k)
        else:
            self._retriever = base_ret
        logger.info("pipeline.ensure_index done")

    @timed("answer")
    def answer(self, question: str) -> AnswerResult:
        logger.info("pipeline.answer question=%s", question)
        if not self._retriever:
            self.ensure_index()
        # Adaptive k (vector-only, no rerank yet)
        adaptive_k_used: int | None = None
        chosen_k_logged: int | None = None
        retriever_for_chain = self._retriever
        if getattr(self.config, "adaptive_k", False):
            try:
                k_attr = None
                if hasattr(self._retriever, "k"):
                    k_attr = getattr(self._retriever, "k")
                    # Temporarily over-fetch up to k_max
                    setattr(self._retriever, "k", self.config.k_max)
                candidate_docs = self._retriever.invoke(question)
                # Restore original k before final retrieval below
                if k_attr is not None:
                    setattr(self._retriever, "k", k_attr)
                texts = [d.page_content for d in candidate_docs]
                if texts:
                    q_vec = self.embedding_provider.embeddings.embed_query(question)
                    doc_vecs = self.embedding_provider.embeddings.embed_documents(texts)
                    sims: list[float] = []
                    for v in doc_vecs:
                        num = sum(a * b for a, b in zip(q_vec, v))
                        den_q = sum(a * a for a in q_vec) ** 0.5
                        den_d = sum(a * a for a in v) ** 0.5
                        sims.append(num / (den_q * den_d + 1e-9))
                    chosen_k = choose_adaptive_k(sims, self.config)
                    adaptive_k_used = chosen_k
                    chosen_k_logged = chosen_k
                    if hasattr(self._retriever, "k"):
                        setattr(self._retriever, "k", chosen_k)
                    retriever_for_chain = self._retriever
            except Exception:  # pragma: no cover
                adaptive_k_used = None
        if chosen_k_logged is None:
            chosen_k_logged = getattr(self._retriever, "k", self.config.top_k)
        chain = self.answer_gen.build_chain(retriever_for_chain)
        t0 = time.time()
        expansion_variants: list[str] | None = None
        original_variant_sources = set()
        expansion_added_sources: list[str] = []
        expansion_recall_gain: bool | None = None
        if getattr(self.config, "query_expansion", False):
            expansions = generate_expansions(question, max_new=self.config.query_expansion_max)
            expansion_variants = list(expansions)
            all_variants = [question] + expansions
            variant_docs: list[list[Document]] = []
            for qv in all_variants:
                try:
                    docs = retriever_for_chain.invoke(qv)
                except Exception:
                    docs = []
                variant_docs.append(docs)
            rank_maps = []
            for docs in variant_docs:
                rmap = {id(doc): r for r, doc in enumerate(docs, start=1)}
                rank_maps.append(rmap)
            combined: dict[int, Document] = {}
            for docs in variant_docs:
                for d in docs:
                    combined[id(d)] = d
            # Track added sources beyond original variant
            if variant_docs:
                original_variant_sources = {d.metadata.get("source_name", d.metadata.get("source", "")) for d in variant_docs[0]}
                all_sources = {d.metadata.get("source_name", d.metadata.get("source", "")) for docs in variant_docs for d in docs}
                expansion_added_sources = sorted(s for s in all_sources - original_variant_sources if s)
                if expansion_added_sources:
                    expansion_recall_gain = True
                else:
                    expansion_recall_gain = False
            fused: list[tuple[float, Document]] = []
            for doc_id, doc in combined.items():
                score = 0.0
                for rm in rank_maps:
                    r = rm.get(doc_id)
                    if r:
                        score += 1.0 / (60 + r)
                fused.append((score, doc))
            fused.sort(key=lambda x: x[0], reverse=True)
            retrieved = [d for _, d in fused[: getattr(self._retriever, "k", self.config.top_k)]]
        else:
            retrieved = retriever_for_chain.invoke(question)
        answer_text = chain.invoke(question)
        # If JSON mode, perform simple citation validation & auto-fill missing citations in supporting_facts
        auto_filled_citations = 0
        if self.config.answer_mode == "json":
            try:
                import json as _json
                parsed = _json.loads(answer_text)
                if isinstance(parsed, dict) and isinstance(parsed.get("supporting_facts"), list):
                    sources_for_fill = [s.metadata.source_name for s in sources_meta] if sources_meta else []
                    primary_citation = sources_for_fill[0] if sources_for_fill else None
                    for fact in parsed["supporting_facts"]:
                        if isinstance(fact, dict):
                            cits = fact.get("citations")
                            if (not cits) and primary_citation:
                                fact["citations"] = [primary_citation]
                                auto_filled_citations += 1
                    # Optionally adjust confidence if we had to auto-fill
                    if auto_filled_citations and isinstance(parsed.get("confidence"), (int, float)):
                        parsed["confidence"] = max(0.0, float(parsed["confidence"])) * 0.9
                    # Attach validator stats
                    parsed.setdefault("_validation", {})["auto_filled_citations"] = auto_filled_citations
                    answer_text = _json.dumps(parsed, ensure_ascii=False)
            except Exception:  # pragma: no cover
                pass
        latency = time.time() - t0
        sources_meta: list[ChunkMetadata] = []
        retrieval_results: list[RetrievalResult] = []
        seen_sources: set[str] = set()
        for d in retrieved:
            md = ChunkMetadata(
                source_name=d.metadata.get("source_name", "Unknown"),
                chunk_id=d.metadata.get("chunk_id", ""),
                page=d.metadata.get("page"),
                section=d.metadata.get("section"),
                section_index=d.metadata.get("section_index"),
                section_level=d.metadata.get("section_level"),
                page_start=d.metadata.get("page_start"),
                page_end=d.metadata.get("page_end"),
                token_count=d.metadata.get("token_count"),
                splitting_mode=d.metadata.get("splitting_mode"),
            )
            retrieval_results.append(RetrievalResult(text=d.page_content, metadata=md))
            if md.source_name not in seen_sources:
                sources_meta.append(md)
                seen_sources.add(md.source_name)
        logger.info("pipeline.answer done sources=%d chunks=%d", len(sources_meta), len(retrieval_results))
        # Append query log line (JSONL)
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            candidate_pool_size = None
            rerank_cache_hit_rate = None
            rerank_cache_hits = None
            if isinstance(self._retriever, RerankingRetriever):
                meta = self._retriever.last_metadata()
                candidate_pool_size = meta.get('candidate_pool_size')
                stats = self._retriever.reranker.last_stats()
                rerank_cache_hit_rate = stats.get('cache_hit_rate')
                rerank_cache_hits = stats.get('cache_hits')
            with (log_dir / "query_log.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "question": question,
                    "latency_sec": round(latency, 3),
                    "retrieval_mode": getattr(self.config, 'retrieval_mode', 'vector'),
                    "retrieval_strategy": self.config.retrieval_strategy,
                    "top_k": self.config.top_k,
                    "adaptive_k_used": adaptive_k_used,
                    "chosen_k": chosen_k_logged,
                    "candidate_pool_size": candidate_pool_size,
                    "rerank_cache_hits": rerank_cache_hits,
                    "rerank_cache_hit_rate": rerank_cache_hit_rate,
                    "chunks": [r.metadata.chunk_id for r in retrieval_results],
                    "query_expansion_used": getattr(self.config, 'query_expansion', False),
                    "expansion_variants": expansion_variants,
                    "sources": [s.source_name for s in sources_meta],
                    "expansion_recall_gain": expansion_recall_gain,
                    "expansion_added_sources": expansion_added_sources[:10] if expansion_added_sources else None,
                    "auto_filled_citations": auto_filled_citations if self.config.answer_mode == "json" else None,
                }) + "\n")
        except Exception:  # pragma: no cover - logging best-effort
            pass
        return AnswerResult(answer=answer_text, sources=sources_meta, raw_chunks=retrieval_results)

    def update_settings(self, **kwargs):
        """Update retrieval-related settings and rebuild the retriever in-place.

        Only affects in-memory retriever (does not trigger re-index). Safe for UI use.
        """
        # Fields that require retriever rebuild when changed
        retriever_fields = {
            "top_k",
            "retrieval_strategy",
            "retrieval_mode",
            "mmr_lambda_mult",
            "mmr_fetch_k_factor",
            "mmr_min_fetch_k",
            "rerank_enable",
            "rerank_model",
            "rerank_overlap_weight",
            "rerank_tfidf_weight",
            "rerank_fetch_k_factor",
            "rerank_fetch_k_max",
        }
        # Fields that only influence logic inside answer() / generator
        light_fields = {
            "adaptive_k",
            "k_min",
            "k_max",
            "query_expansion",
            "query_expansion_max",
            "answer_mode",
            "prompt_version",
        }
        retriever_needs_rebuild = False
        answergen_needs_rebuild = False
        changed_any = False
        for k, v in kwargs.items():
            if not hasattr(self.config, k):
                continue
            current = getattr(self.config, k)
            if current == v:
                continue
            setattr(self.config, k, v)
            changed_any = True
            if k in retriever_fields:
                retriever_needs_rebuild = True
            if k in {"answer_mode", "prompt_version"}:
                answergen_needs_rebuild = True
        # Rebuild retriever if required
        if retriever_needs_rebuild and self.vs_manager.vectorstore:
            if not self._retriever_factory:
                self._retriever_factory = RetrieverFactory(self.vs_manager.vectorstore)
            base_ret = self._retriever_factory.build(self.config)
            if getattr(self.config, 'rerank_enable', False):
                fetch_k = min(self.config.top_k * self.config.rerank_fetch_k_factor, self.config.rerank_fetch_k_max)
                reranker = SimpleReranker(
                    self.config.rerank_model,
                    cache_max=self.config.rerank_cache_max,
                    overlap_weight=self.config.rerank_overlap_weight,
                    tfidf_weight=self.config.rerank_tfidf_weight,
                )
                self._retriever = RerankingRetriever(base_ret, reranker, self.config.top_k, fetch_k=fetch_k)
            else:
                self._retriever = base_ret
            logger.debug("pipeline.update_settings retriever_rebuilt fields=%s", retriever_fields.intersection(kwargs.keys()))
        # Rebuild answer generator (prompt selection) if needed
        if answergen_needs_rebuild:
            self.answer_gen = AnswerGenerator(self.config)
            logger.debug("pipeline.update_settings answer_generator_rebuilt answer_mode=%s prompt_version=%s", self.config.answer_mode, self.config.prompt_version)
        return changed_any
