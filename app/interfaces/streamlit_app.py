from __future__ import annotations
from app.core import RAGPipeline, RAGConfig
from pathlib import Path
import json
import streamlit as st

st.set_page_config(
    page_title="Chat mit PDFs",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }

    /* Settings tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
        color: #1f1f1f;
        font-weight: 500;
    }

    /* Dark mode support for tabs */
    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab"] {
            background-color: #262730;
            color: #fafafa;
        }
    }

    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
        font-weight: 600;
    }

    /* Compact expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .status-active {
        background: #d4edda;
        color: #155724;
    }

    .status-inactive {
        background: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_pipeline():
    pipeline = RAGPipeline()
    pipeline.ensure_index()
    return pipeline

def render_header(pipeline):
    """Render modern header with key stats"""
    docs = pipeline.vs_manager.list_sources()
    doc_count = len(docs) if docs else 0

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("# 🔬 PDF Research Assistant")
        st.caption("Wissenschaftliche Dokumente intelligent durchsuchen")
    with col2:
        st.metric("Indexierte Dokumente", doc_count, delta=None)
    with col3:
        if st.button("🔄 Neuer Chat", use_container_width=True, type="secondary"):
            st.session_state.pop("messages", None)
            st.session_state.pop("last_chunks", None)
            st.session_state.pop("last_telemetry", None)
            st.rerun()

def render_sidebar_settings(pipeline):
    """Render organized sidebar with tabbed settings"""
    with st.sidebar:
        st.markdown("## ⚙️ Konfiguration")

        # Document management section
        with st.expander("📚 Dokumentenverwaltung", expanded=True):
            docs = pipeline.vs_manager.list_sources()
            if docs:
                st.markdown(f"**{len(docs)} Dokument(e) im Index:**")
                # Show first 5 docs with option to see all
                display_docs = docs[:5]
                for d in display_docs:
                    st.markdown(f"• {d}")
                if len(docs) > 5:
                    with st.expander(f"➕ {len(docs) - 5} weitere anzeigen"):
                        for d in docs[5:]:
                            st.markdown(f"• {d}")
            else:
                st.info("💡 Keine Dokumente indexiert")

            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🗑️ Reset", use_container_width=True, type="secondary"):
                    pipeline.vs_manager.reset()
                    st.session_state.pop("messages", None)
                    st.session_state.pop("last_chunks", None)
                    st.success("✓ Vectorstore gelöscht")
                    st.rerun()
            with col_b:
                if st.button("🔨 Rebuild", use_container_width=True, type="primary"):
                    pipeline.ensure_index(force=True)
                    st.success("✓ Index neu aufgebaut")
                    st.rerun()

        st.markdown("---")

        # Tabbed settings for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Retrieval", "🔧 Advanced", "📊 Output", "🧪 Debug"])

        with tab1:
            render_retrieval_settings(pipeline)

        with tab2:
            render_advanced_settings(pipeline)

        with tab3:
            render_output_settings(pipeline)

        with tab4:
            render_debug_settings(pipeline)

        # Apply all changes
        apply_settings_changes(pipeline)

def render_retrieval_settings(pipeline):
    """Core retrieval settings - clean and organized"""
    st.markdown("#### Basis-Einstellungen")

    # Retrieval mode with visual indicators
    retrieval_mode = st.selectbox(
        "🔍 Such-Modus",
        ["vector", "keyword", "hybrid"],
        index=["vector", "keyword", "hybrid"].index(pipeline.config.retrieval_mode),
        help="**Vector**: Semantische Suche | **Keyword**: BM25 | **Hybrid**: RRF Fusion",
        key="retrieval_mode"
    )

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.number_input(
            "📊 Top K",
            min_value=3,
            max_value=50,
            value=pipeline.config.top_k,
            help="Anzahl abzurufender Dokumente",
            key="top_k"
        )
    with col2:
        strategy = st.selectbox(
            "🎲 Strategie",
            ["similarity", "mmr"],
            index=0 if pipeline.config.retrieval_strategy=="similarity" else 1,
            help="**Similarity**: Relevanz | **MMR**: Diversität",
            key="strategy"
        )

    # MMR settings (conditional)
    if strategy == "mmr":
        with st.expander("🎛️ MMR Feintuning"):
            mmr_lambda = st.slider(
                "Lambda (Relevanz ↔ Diversität)",
                0.0, 1.0,
                float(pipeline.config.mmr_lambda_mult),
                0.05,
                help="1.0 = reine Relevanz, 0.0 = maximale Diversität",
                key="mmr_lambda"
            )
            col1, col2 = st.columns(2)
            with col1:
                mmr_fetch_factor = st.number_input(
                    "Fetch Faktor",
                    1, 10,
                    int(pipeline.config.mmr_fetch_k_factor),
                    key="mmr_fetch_factor"
                )
            with col2:
                mmr_min_fetch = st.number_input(
                    "Min. Fetch",
                    10, 200,
                    int(pipeline.config.mmr_min_fetch_k),
                    step=10,
                    key="mmr_min_fetch"
                )
            st.session_state.mmr_settings = {
                "mmr_lambda_mult": mmr_lambda,
                "mmr_fetch_k_factor": mmr_fetch_factor,
                "mmr_min_fetch_k": mmr_min_fetch
            }

    # Adaptive k
    st.markdown("#### Adaptive Retrieval")
    adaptive_k = st.toggle(
        "🎯 Adaptive K aktivieren",
        value=pipeline.config.adaptive_k,
        help="Dynamische Anpassung basierend auf Score-Verteilung",
        key="adaptive_k"
    )

    if adaptive_k:
        col1, col2 = st.columns(2)
        with col1:
            k_min = st.number_input("Min K", 1, 50, pipeline.config.k_min, key="k_min")
        with col2:
            k_max = st.number_input("Max K", 1, 100, pipeline.config.k_max, key="k_max")
        st.session_state.adaptive_k_settings = {"k_min": k_min, "k_max": k_max}

def render_advanced_settings(pipeline):
    """Advanced features: reranking, expansion, chunking"""

    # Chunking settings (requires rebuild)
    st.markdown("#### 📄 Chunking")
    chunking_mode = st.selectbox(
        "Modus",
        ["basic", "structure"],
        index=["basic", "structure"].index(pipeline.config.chunking_mode),
        help="**Structure**: Nutzt Dokumentstruktur für bessere Chunks",
        key="chunking_mode"
    )
    accurate_token_count = st.toggle(
        "Exakte Token-Zählung",
        value=pipeline.config.accurate_token_count,
        help="Verwendet tiktoken (benötigt Rebuild)",
        key="accurate_token_count"
    )

    # Check if rebuild needed
    rebuild_needed = (
        chunking_mode != pipeline.config.chunking_mode or
        accurate_token_count != pipeline.config.accurate_token_count
    )
    if rebuild_needed:
        st.warning("⚠️ Änderungen erfordern Index-Rebuild")
        if st.button("🔨 Jetzt neu bauen", use_container_width=True):
            pipeline.config.chunking_mode = chunking_mode
            pipeline.config.accurate_token_count = accurate_token_count
            pipeline.ensure_index(force=True)
            st.success("✓ Index neu aufgebaut")
            st.rerun()

    st.markdown("---")

    # Reranking
    st.markdown("#### 🎯 Reranking")
    rerank_enable = st.toggle(
        "Reranking aktivieren",
        value=pipeline.config.rerank_enable,
        help="Heuristisches Re-Ranking Layer",
        key="rerank_enable"
    )

    if rerank_enable:
        col1, col2 = st.columns(2)
        with col1:
            rr_overlap_w = st.number_input(
                "Overlap Weight",
                0.0, 5.0,
                float(pipeline.config.rerank_overlap_weight),
                0.1,
                key="rr_overlap_w"
            )
        with col2:
            rr_tfidf_w = st.number_input(
                "TF-IDF Weight",
                0.0, 5.0,
                float(pipeline.config.rerank_tfidf_weight),
                0.1,
                key="rr_tfidf_w"
            )

        # Extended reranker settings
        with st.expander("⚙️ Erweiterte Reranker-Settings"):
            rr_fetch_factor = getattr(pipeline.config, "rerank_fetch_k_factor", None)
            rr_fetch_max = getattr(pipeline.config, "rerank_fetch_k_max", None)
            rr_cache_max = getattr(pipeline.config, "rerank_cache_max", None)

            if rr_fetch_factor is not None:
                rr_fetch_factor = st.number_input(
                    "Fetch Faktor", 1, 20, int(rr_fetch_factor),
                    help="Kandidaten-Überfetch Multiplikator",
                    key="rr_fetch_factor"
                )
            if rr_fetch_max is not None:
                rr_fetch_max = st.number_input(
                    "Fetch Max", 1, 500, int(rr_fetch_max),
                    help="Max. Kandidaten für Reranker",
                    key="rr_fetch_max"
                )
            if rr_cache_max is not None:
                rr_cache_max = st.number_input(
                    "Cache Max", 0, 20000, int(rr_cache_max),
                    help="LRU Cache Größe (0 = aus)",
                    key="rr_cache_max"
                )

            st.session_state.rerank_extended = {
                "rerank_fetch_k_factor": rr_fetch_factor,
                "rerank_fetch_k_max": rr_fetch_max,
                "rerank_cache_max": rr_cache_max
            }

        st.session_state.rerank_settings = {
            "rerank_overlap_weight": rr_overlap_w,
            "rerank_tfidf_weight": rr_tfidf_w
        }

    st.markdown("---")

    # Query Expansion
    st.markdown("#### 🔄 Query Expansion")
    query_expansion = st.toggle(
        "Query Expansion aktivieren",
        value=pipeline.config.query_expansion,
        help="Synonyme & Zerlegung via RRF",
        key="query_expansion"
    )

    if query_expansion:
        expansion_max = st.slider(
            "Max. Expansion Varianten",
            0, 5,
            pipeline.config.query_expansion_max,
            help="Zusätzliche Varianten neben Original",
            key="expansion_max"
        )
        st.session_state.expansion_settings = {"query_expansion_max": expansion_max}

def render_output_settings(pipeline):
    """Output format and prompt settings"""
    st.markdown("#### 📝 Antwortformat")

    answer_mode = st.selectbox(
        "Format",
        ["text", "json"],
        index=["text", "json"].index(pipeline.config.answer_mode),
        help="**Text**: Natürlicher Text | **JSON**: Strukturierte Ausgabe",
        key="answer_mode"
    )

    prompt_version = st.selectbox(
        "Prompt Version",
        ["v1", "v2", "v3_json"],
        index=["v1","v2","v3_json"].index(pipeline.config.prompt_version),
        help="Wähle Prompt-Template Version",
        key="prompt_version"
    )

    # Vector backend (if available)
    vector_backend = getattr(pipeline.config, "vector_backend", None)
    if vector_backend is not None:
        st.markdown("---")
        st.markdown("#### 🗄️ Vector Backend")
        vector_backend = st.selectbox(
            "Backend",
            ["chroma", "faiss", "milvus"],
            index=["chroma", "faiss", "milvus"].index(vector_backend) if vector_backend in ("chroma","faiss","milvus") else 0,
            help="Vector-Datenbank (erfordert Rebuild)",
            key="vector_backend"
        )
        st.session_state.vector_backend = vector_backend

def render_debug_settings(pipeline):
    """Debug and evaluation tools"""
    st.markdown("#### 🤖 LLM Modell")

    # Model selector
    available_models = ["gemini-2.5-flash", "gemini-2.5-pro"]
    current_model = pipeline.config.llm_model
    # Default to flash if current model is not in the list
    default_index = available_models.index(current_model) if current_model in available_models else 0

    llm_model = st.selectbox(
        "Gemini Modell",
        available_models,
        index=default_index,
        help="**Flash**: Schneller, kostengünstiger – Optimal für die meisten Anfragen\n**Pro**: Leistungsstärker, bessere Reasoning-Fähigkeiten für komplexe Analysen",
        key="llm_model"
    )

    # Show model info
    if "flash" in llm_model:
        st.caption("⚡ Schnelles Modell – Optimal für die meisten Anfragen")
    else:
        st.caption("🧠 Leistungsstarkes Modell – Bessere Analyse komplexer Fragen")

    st.markdown("---")
    st.markdown("#### 🧪 Entwickler-Tools")

    if hasattr(pipeline, "run_eval"):
        if st.button("▶️ Mini-Evaluation ausführen", use_container_width=True):
            with st.spinner("Evaluation läuft..."):
                res = pipeline.run_eval(sample_only=True)
            st.json(res)

    st.markdown("---")
    st.markdown("#### 📊 Aktuelle Konfiguration")

    with st.expander("Vollständige Config anzeigen"):
        config_dict = {
            k: v for k, v in vars(pipeline.config).items()
            if not k.startswith('_')
        }
        st.json(config_dict, expanded=False)

def apply_settings_changes(pipeline):
    """Collect and apply all settings changes"""
    update_kwargs = {}

    # Basic retrieval
    if hasattr(st.session_state, "retrieval_mode") and st.session_state.retrieval_mode != pipeline.config.retrieval_mode:
        update_kwargs["retrieval_mode"] = st.session_state.retrieval_mode
    if hasattr(st.session_state, "top_k") and st.session_state.top_k != pipeline.config.top_k:
        update_kwargs["top_k"] = st.session_state.top_k
    if hasattr(st.session_state, "strategy") and st.session_state.strategy != pipeline.config.retrieval_strategy:
        update_kwargs["retrieval_strategy"] = st.session_state.strategy

    # MMR settings
    if hasattr(st.session_state, "mmr_settings"):
        update_kwargs.update(st.session_state.mmr_settings)

    # Adaptive k
    if hasattr(st.session_state, "adaptive_k") and st.session_state.adaptive_k != pipeline.config.adaptive_k:
        update_kwargs["adaptive_k"] = st.session_state.adaptive_k
    if hasattr(st.session_state, "adaptive_k_settings"):
        update_kwargs.update(st.session_state.adaptive_k_settings)

    # Reranking
    if hasattr(st.session_state, "rerank_enable") and st.session_state.rerank_enable != pipeline.config.rerank_enable:
        update_kwargs["rerank_enable"] = st.session_state.rerank_enable
    if hasattr(st.session_state, "rerank_settings"):
        update_kwargs.update(st.session_state.rerank_settings)
    if hasattr(st.session_state, "rerank_extended"):
        for k, v in st.session_state.rerank_extended.items():
            if v is not None:
                update_kwargs[k] = int(v)

    # Query expansion
    if hasattr(st.session_state, "query_expansion") and st.session_state.query_expansion != pipeline.config.query_expansion:
        update_kwargs["query_expansion"] = st.session_state.query_expansion
    if hasattr(st.session_state, "expansion_settings"):
        update_kwargs.update(st.session_state.expansion_settings)

    # Output settings
    if hasattr(st.session_state, "answer_mode") and st.session_state.answer_mode != pipeline.config.answer_mode:
        update_kwargs["answer_mode"] = st.session_state.answer_mode
    if hasattr(st.session_state, "prompt_version") and st.session_state.prompt_version != pipeline.config.prompt_version:
        update_kwargs["prompt_version"] = st.session_state.prompt_version

    # LLM Model
    if hasattr(st.session_state, "llm_model") and st.session_state.llm_model != pipeline.config.llm_model:
        update_kwargs["llm_model"] = st.session_state.llm_model

    # Apply updates
    if update_kwargs:
        pipeline.update_settings(**update_kwargs)

def render_telemetry(telemetry):
    """Render telemetry in modern card layout"""
    if not telemetry:
        return

    st.markdown("### 📊 Query Telemetrie")

    # Primary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chosen K", telemetry.get("chosen_k", "—"))
    col2.metric("Adaptive K", "✓" if telemetry.get("adaptive_k_used") else "—")
    col3.metric("Expansion", "✓" if telemetry.get("query_expansion_used") else "—")
    col4.metric("Pool Size", telemetry.get("candidate_pool_size", "—"))

    # Cache metrics if available
    if telemetry.get("rerank_cache_hits") is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Cache Hits", telemetry.get("rerank_cache_hits", 0))
        col2.metric("Hit Rate", telemetry.get("rerank_cache_hit_rate", "—"))
        col3.metric("Spacer", "", label_visibility="hidden")  # Spacer

    # Expansion details
    if telemetry.get("expansion_added_sources"):
        with st.expander("🔍 Expansion Details"):
            sources = telemetry.get("expansion_added_sources", [])
            st.markdown("**Hinzugefügte Quellen:**")
            for src in sources:
                st.markdown(f"• {src}")

def render_chunks(chunks):
    """Render retrieved chunks in modern card style"""
    if not chunks:
        return

    st.markdown("### 📚 Abgerufene Chunks")

    for idx, rr in enumerate(chunks, 1):
        meta = rr.metadata

        # Build header
        header = f"**{idx}. {meta.source_name}**"
        badges = []

        if meta.section:
            badges.append(f"📑 {meta.section}")
        if meta.page_start or meta.page:
            page_info = f"S. {meta.page_start or meta.page}"
            if meta.page_end and meta.page_end != meta.page_start:
                page_info += f"-{meta.page_end}"
            badges.append(page_info)
        if meta.token_count:
            badges.append(f"🔤 {meta.token_count} tokens")

        with st.expander(f"{header} — {' • '.join(badges)}"):
            # Additional metadata
            details = []
            if meta.section_level:
                details.append(f"Level {meta.section_level}")
            if meta.section_index:
                details.append(f"Index {meta.section_index}")
            if meta.splitting_mode:
                details.append(f"Mode: {meta.splitting_mode}")

            if details:
                st.caption(" | ".join(details))

            # Content preview
            content = rr.text
            if len(content) > 1000:
                st.markdown(content[:1000] + "...")
                with st.expander("➕ Vollständigen Chunk anzeigen"):
                    st.text(content)
            else:
                st.markdown(content)

def main():
    pipeline = init_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render header
    render_header(pipeline)

    # Render sidebar
    render_sidebar_settings(pipeline)

    # Main chat interface with improved styling
    st.markdown("---")

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("💬 Stelle eine Frage zu deinen PDFs..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Durchsuche Dokumente..."):
                result = pipeline.answer(prompt)

            # Parse answer based on mode
            if pipeline.config.answer_mode == "json":
                try:
                    parsed = json.loads(result.answer)
                    answer_display = parsed.get("direct_answer") or result.answer
                except Exception:
                    answer_display = result.answer
            else:
                answer_display = result.answer

            # Format sources nicely
            sources_text = "\n\n---\n\n**📚 Quellen:**\n"
            for idx, s in enumerate(result.sources, 1):
                sources_text += f"{idx}. {s.source_name}\n"

            full_answer = answer_display + sources_text
            st.markdown(full_answer)

        # Store assistant message
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
        st.session_state.last_chunks = result.raw_chunks

        # Read telemetry (best effort)
        try:
            log_path = Path("logs/query_log.jsonl")
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8").strip().splitlines()
                if lines:
                    telemetry = json.loads(lines[-1])
                    st.session_state.last_telemetry = telemetry
        except Exception:
            pass

        st.rerun()

    # Show telemetry and chunks in expandable sections (below chat)
    if "last_chunks" in st.session_state or "last_telemetry" in st.session_state:
        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            if "last_telemetry" in st.session_state:
                with st.expander("📊 Query Telemetrie", expanded=False):
                    render_telemetry(st.session_state.last_telemetry)

        with col2:
            if "last_chunks" in st.session_state:
                with st.expander(f"📚 Abgerufene Chunks ({len(st.session_state.last_chunks)})", expanded=False):
                    render_chunks(st.session_state.last_chunks)

if __name__ == "__main__":
    main()
