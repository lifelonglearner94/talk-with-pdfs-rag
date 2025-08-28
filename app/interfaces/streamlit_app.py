from __future__ import annotations
from app.core import RAGPipeline, RAGConfig
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
        top_k = st.slider("Top K", 3, 25, pipeline.config.top_k)
        strategy = st.selectbox("Strategie", ["similarity", "mmr"], index=0 if pipeline.config.retrieval_strategy=="similarity" else 1)
        # Show MMR tuning only if selected
        mmr_lambda = pipeline.config.mmr_lambda_mult
        mmr_fetch_factor = pipeline.config.mmr_fetch_k_factor
        mmr_min_fetch = pipeline.config.mmr_min_fetch_k
        if strategy == "mmr":
            st.caption("MMR Diversitätseinstellungen")
            mmr_lambda = st.slider("Lambda (Relevanz↔Diversität)", 0.0, 1.0, float(mmr_lambda), 0.05, help="1.0 = reine Relevanz, 0 = maximale Diversität")
            mmr_fetch_factor = st.slider("Fetch k Faktor", 1, 10, int(mmr_fetch_factor), help="Kandidaten-Multiplikator: fetch_k = max(k * Faktor, Mindest-Fetch k)")
            mmr_min_fetch = st.slider("Mindest Fetch k", 10, 200, int(mmr_min_fetch), 5, help="Untergrenze für Kandidatenpool")
        if top_k != pipeline.config.top_k or strategy != pipeline.config.retrieval_strategy:
            pipeline.config.top_k = top_k
            pipeline.config.retrieval_strategy = strategy
            pipeline._retriever = pipeline.vs_manager.as_retriever(pipeline.config)
        # Apply MMR param changes (only rebuild retriever if strategy is mmr and values changed)
        if strategy == "mmr" and (
            mmr_lambda != pipeline.config.mmr_lambda_mult or
            mmr_fetch_factor != pipeline.config.mmr_fetch_k_factor or
            mmr_min_fetch != pipeline.config.mmr_min_fetch_k
        ):
            pipeline.config.mmr_lambda_mult = mmr_lambda
            pipeline.config.mmr_fetch_k_factor = mmr_fetch_factor
            pipeline.config.mmr_min_fetch_k = mmr_min_fetch
            pipeline._retriever = pipeline.vs_manager.as_retriever(pipeline.config)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Frage stellen..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.spinner("Suche und generiere Antwort..."):
            result = pipeline.answer(prompt)
        answer_text = result.answer + "\n\n**Quellen:**\n" + "\n".join(f"- {s.source_name}" for s in result.sources)
        st.session_state.messages.append({"role":"assistant","content":answer_text})
        st.session_state.last_chunks = result.raw_chunks
        st.rerun()

    if "last_chunks" in st.session_state:
        with st.expander("Abrufene Chunks"):
            for rr in st.session_state.last_chunks:
                st.markdown(f"**{rr.metadata.source_name}** ({rr.metadata.chunk_id})")
                st.code(rr.text[:800] + ("..." if len(rr.text) > 800 else ""))

if __name__ == "__main__":
    main()
