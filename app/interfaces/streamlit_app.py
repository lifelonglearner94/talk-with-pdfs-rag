from __future__ import annotations
from app.core import RAGPipeline, RAGConfig
from pathlib import Path
import json
import streamlit as st

st.set_page_config(page_title="Chat mit PDFs", page_icon="🔬", layout="wide", initial_sidebar_state="collapsed")

@st.cache_resource
def init_pipeline():
    pipeline = RAGPipeline()
    pipeline.ensure_index()
    return pipeline

def main():
    st.title("🔬 Chat mit wissenschaftlichen PDFs")
    pipeline = init_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar controls
    with st.sidebar:
        st.header("Einstellungen")
        # Current indexed documents
        with st.expander("📄 Dokumente im Index", expanded=True):
            docs = pipeline.vs_manager.list_sources()
            if docs:
                st.markdown("\n".join(f"- {d}" for d in docs))
                st.caption(f"{len(docs)} Dokument(e)")
            else:
                st.info("Noch keine Vektoren aufgebaut oder leer.")

        # Reset / Rebuild controls
        with st.expander("⚠️ Reset & Rebuild", expanded=False):
            st.caption("Komplett zurücksetzen löscht den Vectorstore.")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Vectorstore löschen", type="secondary"):
                    pipeline.vs_manager.reset()
                    st.session_state.pop("messages", None)
                    st.session_state.pop("last_chunks", None)
                    st.success("Vectorstore gelöscht. Baue beim nächsten Zugriff neu auf.")
                    st.experimental_rerun()
            with col_b:
                if st.button("Neu indexieren", type="primary"):
                    pipeline.ensure_index()  # will rebuild if needed
                    st.success("Index aktualisiert.")
                    st.experimental_rerun()
        st.markdown("---")
        st.subheader("Retrieval")
        st.subheader("Chunking")
        chunking_mode = st.selectbox(
            "Chunking Modus",
            ["basic", "structure"],
            index=["basic", "structure"].index(pipeline.config.chunking_mode),
            help="Strukturierter Modus nutzt Abschnittserkennung & Metadaten.",
        )
        accurate_token_count = st.checkbox(
            "Exakte Tokenzählung (tiktoken)",
            value=pipeline.config.accurate_token_count,
            help="Kann Index-Rebuild erfordern (Flag wirkt beim nächsten Rebuild).",
        )
        retrieval_mode = st.selectbox(
            "Modus",
            ["vector", "keyword", "hybrid"],
            index=["vector", "keyword", "hybrid"].index(pipeline.config.retrieval_mode),
            help="Wähle reine Vektorsuche, reine BM25 Schlüsselwortsuche oder Hybrid (RRF Fusion).",
        )
        top_k = st.slider("Top K", 3, 50, pipeline.config.top_k)
        strategy = st.selectbox("Strategie", ["similarity", "mmr"], index=0 if pipeline.config.retrieval_strategy=="similarity" else 1, help="Vector Retrieval Strategie")
        # Show MMR tuning only if selected
        mmr_lambda = pipeline.config.mmr_lambda_mult
        mmr_fetch_factor = pipeline.config.mmr_fetch_k_factor
        mmr_min_fetch = pipeline.config.mmr_min_fetch_k
        if strategy == "mmr":
            st.caption("MMR Diversitätseinstellungen")
            mmr_lambda = st.slider("Lambda (Relevanz↔Diversität)", 0.0, 1.0, float(mmr_lambda), 0.05, help="1.0 = reine Relevanz, 0 = maximale Diversität")
            mmr_fetch_factor = st.slider("Fetch k Faktor", 1, 10, int(mmr_fetch_factor), help="Kandidaten-Multiplikator: fetch_k = max(k * Faktor, Mindest-Fetch k)")
            mmr_min_fetch = st.slider("Mindest Fetch k", 10, 200, int(mmr_min_fetch), 5, help="Untergrenze für Kandidatenpool")
        st.markdown("---")
        st.subheader("Reranking & Expansion")
        rerank_enable = st.checkbox("Reranking aktivieren", value=pipeline.config.rerank_enable, help="Aktiviere heuristisches Reranking Layer")
        col_rr1, col_rr2 = st.columns(2)
        with col_rr1:
            rr_overlap_w = st.number_input("Overlap Gewicht", 0.0, 5.0, float(pipeline.config.rerank_overlap_weight), 0.1, help="Gewicht für unique Query Token Overlap")
        with col_rr2:
            rr_tfidf_w = st.number_input("TF-IDF Gewicht", 0.0, 5.0, float(pipeline.config.rerank_tfidf_weight), 0.1, help="Gewicht für einfachen TF-IDF Score")
        # Additional reranker knobs (may not exist on older pipelines)
        rr_fetch_factor = getattr(pipeline.config, "rerank_fetch_k_factor", None)
        rr_fetch_max = getattr(pipeline.config, "rerank_fetch_k_max", None)
        rr_cache_max = getattr(pipeline.config, "rerank_cache_max", None)
        if rr_fetch_factor is not None or rr_fetch_max is not None or rr_cache_max is not None:
            st.caption("Erweiterte Reranker Einstellungen")
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                if rr_fetch_factor is not None:
                    rr_fetch_factor = st.number_input("Rerank Fetch Faktor", 1, 20, int(rr_fetch_factor), help="Multiplikator für Kandidaten-Überfetch vor Rerank")
            with col_r2:
                if rr_fetch_max is not None:
                    rr_fetch_max = st.number_input("Rerank Fetch Max", 1, 500, int(rr_fetch_max), help="Maximale Anzahl Kandidaten, die an den Reranker geschickt werden")
            with col_r3:
                if rr_cache_max is not None:
                    rr_cache_max = st.number_input("Rerank Cache Max", 0, 20000, int(rr_cache_max), help="Max Einträge im LRU Rerank Cache (0 = deaktiviert)")
        query_expansion = st.checkbox("Heuristische Query Expansion", value=pipeline.config.query_expansion, help="Synonyme & Zerlegung kombinieren via RRF")
        expansion_max = st.slider("Max. Expansion Varianten", 0, 5, pipeline.config.query_expansion_max, help="Zusätzliche Varianten außer Original")
        st.markdown("---")
        st.subheader("Adaptive k")
        adaptive_k = st.checkbox("Adaptive k aktivieren", value=pipeline.config.adaptive_k, help="Dynamische Auswahl basierend auf Score Streuung")
        col_ak1, col_ak2 = st.columns(2)
        with col_ak1:
            k_min = st.number_input("k_min", 1, 50, pipeline.config.k_min)
        with col_ak2:
            k_max = st.number_input("k_max", 1, 100, pipeline.config.k_max)
        st.markdown("---")
        st.subheader("Antwortformat")
        answer_mode = st.selectbox("Answer Mode", ["text", "json"], index=["text", "json"].index(pipeline.config.answer_mode))
        prompt_version = st.selectbox("Prompt Version", ["v1", "v2", "v3_json"], index=["v1","v2","v3_json"].index(pipeline.config.prompt_version))
        # Vector backend selection (if pipeline exposes it)
        vector_backend = getattr(pipeline.config, "vector_backend", None)
        if vector_backend is not None:
            vector_backend = st.selectbox("Vector Backend", ["chroma", "faiss", "milvus"], index=["chroma", "faiss", "milvus"].index(vector_backend) if vector_backend in ("chroma","faiss","milvus") else 0, help="Wähle das Vector-Backend (nur für Anzeige / Rebuild relevant)")
        # Apply changes via pipeline encapsulation
        update_kwargs = {}
        if retrieval_mode != pipeline.config.retrieval_mode:
            update_kwargs["retrieval_mode"] = retrieval_mode
        if top_k != pipeline.config.top_k:
            update_kwargs["top_k"] = top_k
        if strategy != pipeline.config.retrieval_strategy:
            update_kwargs["retrieval_strategy"] = strategy
        if strategy == "mmr":
            if mmr_lambda != pipeline.config.mmr_lambda_mult:
                update_kwargs["mmr_lambda_mult"] = mmr_lambda
            if mmr_fetch_factor != pipeline.config.mmr_fetch_k_factor:
                update_kwargs["mmr_fetch_k_factor"] = mmr_fetch_factor
            if mmr_min_fetch != pipeline.config.mmr_min_fetch_k:
                update_kwargs["mmr_min_fetch_k"] = mmr_min_fetch
        if rerank_enable != pipeline.config.rerank_enable:
            update_kwargs["rerank_enable"] = rerank_enable
        if rr_overlap_w != pipeline.config.rerank_overlap_weight:
            update_kwargs["rerank_overlap_weight"] = rr_overlap_w
        if rr_tfidf_w != pipeline.config.rerank_tfidf_weight:
            update_kwargs["rerank_tfidf_weight"] = rr_tfidf_w
        if query_expansion != pipeline.config.query_expansion:
            update_kwargs["query_expansion"] = query_expansion
        if expansion_max != pipeline.config.query_expansion_max:
            update_kwargs["query_expansion_max"] = expansion_max
        if adaptive_k != pipeline.config.adaptive_k:
            update_kwargs["adaptive_k"] = adaptive_k
        if k_min != pipeline.config.k_min:
            update_kwargs["k_min"] = k_min
        if k_max != pipeline.config.k_max:
            update_kwargs["k_max"] = k_max
        if answer_mode != pipeline.config.answer_mode:
            update_kwargs["answer_mode"] = answer_mode
        if prompt_version != pipeline.config.prompt_version:
            update_kwargs["prompt_version"] = prompt_version
        if update_kwargs:
            pipeline.update_settings(**update_kwargs)
        # Non-dynamic fields needing full rebuild (chunking mode / accurate token count)
        rebuild_needed = False
        if chunking_mode != pipeline.config.chunking_mode:
            pipeline.config.chunking_mode = chunking_mode
            rebuild_needed = True
        if accurate_token_count != pipeline.config.accurate_token_count:
            pipeline.config.accurate_token_count = accurate_token_count
            rebuild_needed = True
        if rebuild_needed and st.button("Index mit neuen Chunking Einstellungen neu bauen"):
            pipeline.ensure_index(force=True)
            st.success("Index neu aufgebaut mit aktualisierten Chunking Parametern.")
            st.experimental_rerun()
        # Optional: run a small eval if pipeline exposes it
        if hasattr(pipeline, "run_eval"):
            if st.button("Kleines Evaluationstest-Run ausführen"):
                with st.spinner("Eval läuft..."):
                    res = pipeline.run_eval(sample_only=True)
                st.json(res)
    # ...existing code...

    # Main-area control: Neuer Chat starten
    if st.button("Neuer Chat", key="new_chat_main"):
        st.session_state.pop("messages", None)
        st.session_state.pop("last_chunks", None)
        st.session_state.pop("last_telemetry", None)
        st.experimental_rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Frage stellen..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.spinner("Suche und generiere Antwort..."):
            result = pipeline.answer(prompt)
        # Display differently if JSON mode
        if pipeline.config.answer_mode == "json":
            try:
                parsed = json.loads(result.answer)
                answer_display = parsed.get("direct_answer") or result.answer
            except Exception:
                answer_display = result.answer
        else:
            answer_display = result.answer
        answer_text = answer_display + "\n\n**Quellen:**\n" + "\n".join(f"- {s.source_name}" for s in result.sources)
        st.session_state.messages.append({"role":"assistant","content":answer_text})
        st.session_state.last_chunks = result.raw_chunks
        # Read last log line for telemetry (best effort)
        try:
            log_path = Path("logs/query_log.jsonl")
            if log_path.exists():
                *_, last = log_path.read_text(encoding="utf-8").strip().splitlines()
                telemetry = json.loads(last)
                st.session_state.last_telemetry = telemetry
        except Exception:
            pass
        st.rerun()

    if "last_chunks" in st.session_state:
        with st.expander("Abgerufene Chunks & Telemetrie"):
            if telemetry := st.session_state.get("last_telemetry"):
                st.markdown("**Query Telemetrie**")
                cols = st.columns(3)
                cols[0].metric("Chosen k", telemetry.get("chosen_k"))
                cols[1].metric("Adaptive?k", str(bool(telemetry.get("adaptive_k_used"))))
                cols[2].metric("Expansion", str(telemetry.get("query_expansion_used")))
                if telemetry.get("candidate_pool_size"):
                    cols2 = st.columns(3)
                    cols2[0].metric("Pool", telemetry.get("candidate_pool_size"))
                    cols2[1].metric("Cache Hits", telemetry.get("rerank_cache_hits"))
                    cols2[2].metric("Hit-Rate", telemetry.get("rerank_cache_hit_rate"))
                if telemetry.get("expansion_added_sources"):
                    st.caption("Expansion Added Sources: " + ", ".join(telemetry.get("expansion_added_sources") or []))
            st.markdown("---")
            for rr in st.session_state.last_chunks:
                meta = rr.metadata
                header = f"**{meta.source_name}** ({meta.chunk_id})"
                details = []
                if meta.section:
                    details.append(f"Abschnitt: {meta.section} (idx {meta.section_index}, lvl {meta.section_level})")
                if meta.page_start or meta.page_end:
                    pr = f"Seiten: {meta.page_start or meta.page}–{meta.page_end or meta.page}" if meta.page_start or meta.page_end else f"Seite: {meta.page}"
                    details.append(pr)
                if meta.token_count:
                    details.append(f"Tokens: {meta.token_count}")
                if meta.splitting_mode:
                    details.append(f"Splitter: {meta.splitting_mode}")
                if details:
                    header += "  ".join([" "] + details)
                st.markdown(header)
                st.code(rr.text[:800] + ("..." if len(rr.text) > 800 else ""))

if __name__ == "__main__":
    main()
