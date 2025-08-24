import streamlit as st
import sys
from pathlib import Path
from main import ScientificPDFSearcher

# Set page config
st.set_page_config(
    page_title="Chat mit PDFs",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling - optimized for both light and dark modes
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        color: var(--text-color);
        margin-bottom: 2rem;
        font-weight: bold;
    }

    /* Chat message containers */
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    /* User message styling - adapts to theme */
    .user-message {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.15), rgba(33, 150, 243, 0.05));
        border-left: 4px solid #2196F3;
        border-radius: 12px;
    }

    /* Assistant message styling - adapts to theme */
    .assistant-message {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(76, 175, 80, 0.05));
        border-left: 4px solid #4CAF50;
        border-radius: 12px;
    }

    /* Dark mode specific styles */
    @media (prefers-color-scheme: dark) {
        .user-message {
            background: linear-gradient(135deg, rgba(33, 150, 243, 0.25), rgba(33, 150, 243, 0.1));
            border-left: 4px solid #42A5F5;
        }

        .assistant-message {
            background: linear-gradient(135deg, rgba(76, 175, 80, 0.25), rgba(76, 175, 80, 0.1));
            border-left: 4px solid #66BB6A;
        }

        .main-header {
            color: #E3F2FD;
        }
    }

    /* Button styling that works in both themes */
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #2196F3, #1976D2);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(33, 150, 243, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(45deg, #1976D2, #1565C0);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(33, 150, 243, 0.4);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(33, 150, 243, 0.3);
    }

    /* Clear chat button styling */
    .stButton > button[key="clear_chat"] {
        background: linear-gradient(45deg, #f44336, #d32f2f);
        box-shadow: 0 2px 4px rgba(244, 67, 54, 0.3);
    }

    .stButton > button[key="clear_chat"]:hover {
        background: linear-gradient(45deg, #d32f2f, #c62828);
        box-shadow: 0 4px 8px rgba(244, 67, 54, 0.4);
    }

    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid transparent;
        background-color: rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #2196F3;
        box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.1);
    }

    /* Sidebar improvements */
    .css-1d391kg {
        background-color: rgba(0, 0, 0, 0.02);
    }

    /* Success/Error message improvements */
    .stSuccess {
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }

    .stError {
        border-radius: 8px;
        border-left: 4px solid #f44336;
    }

    .stInfo {
        border-radius: 8px;
        border-left: 4px solid #2196F3;
    }

    /* Loading spinner improvements */
    .stSpinner > div {
        border-color: #2196F3 transparent transparent transparent;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .chat-message {
            padding: 0.8rem;
            margin: 0.5rem 0;
        }

        .main-header {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_searcher():
    """PDF-Sucher mit Caching initialisieren"""
    try:
        searcher = ScientificPDFSearcher()
        searcher.load_or_create_vectorstore()
        return searcher, None
    except Exception as e:
        return None, str(e)

def main():
    # Header with improved styling
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Chat mit Ihren wissenschaftlichen PDFs</h1>
        <p style="opacity: 0.8; font-size: 1.1rem; margin-top: -0.5rem;">
            Stellen Sie Fragen zu Ihren Forschungsarbeiten und erhalten Sie KI-gestützte Einblicke
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Initialize searcher
    with st.spinner("🔄 PDF-Suchsystem wird initialisiert..."):
        searcher, error = initialize_searcher()

    if error:
        st.error(f"❌ Fehler beim Initialisieren des Systems: {error}")
        st.info("💡 Stellen Sie sicher, dass Sie PDF-Dateien im 'data'-Ordner haben und Ihr GOOGLE_API_KEY in einer .env-Datei gesetzt ist")
        return

    if not searcher:
        st.error("Fehler beim Initialisieren des Suchsystems")
        return

    st.success("✅ System bereit! Sie können jetzt Fragen zu Ihren PDFs stellen.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history using native Streamlit chat elements
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message["content"])

    # Chat input using Streamlit's chat input (better for mobile and themes)
    if user_input := st.chat_input("Stellen Sie eine Frage zu Ihren PDFs..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get response from the searcher
        with st.spinner("🔍 Suche nach relevanten Informationen..."):
            try:
                response = searcher.search(user_input)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Rerun to update the display
                st.rerun()

            except Exception as e:
                st.error(f"❌ Fehler beim Verarbeiten Ihrer Frage: {str(e)}")
                # Remove the user message since we couldn't process it
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    st.session_state.messages.pop()

    # Clear chat button with better spacing
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🗑️ Chat löschen", key="clear_chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()

    # Sidebar with information
    with st.sidebar:
        st.markdown("### 📚 Über dieses System")
        st.markdown("""
        Dieses intelligente System ermöglicht es Ihnen, Gespräche mit Ihren wissenschaftlichen PDFs zu führen, unterstützt von fortschrittlicher KI.

        #### 🔄 So funktioniert es:
        1. **📖 Verarbeitung**: PDFs werden analysiert und indexiert
        2. **🔍 Abgleich**: Fragen werden mit relevantem Inhalt abgeglichen
        3. **🤖 KI-Antwort**: Gemini liefert detaillierte Antworten

        #### 💡 Tipps für beste Ergebnisse:
        - ✨ Stellen Sie spezifische, detaillierte Fragen
        - 🔬 Verwenden Sie wissenschaftliche Terminologie aus Ihrem Fachgebiet
        - 📊 Fragen Sie nach Methoden, Ergebnissen oder Schlussfolgerungen
        - 🎯 Beziehen Sie sich auf spezifische Konzepte oder Erkenntnisse
        """)

        # Show available PDFs with better formatting
        data_dir = Path("data")
        if data_dir.exists():
            pdf_files = [f.name for f in data_dir.glob("*.pdf") if ":Zone.Identifier" not in f.name]
            if pdf_files:
                st.markdown("### 📄 Verfügbare Dokumente")
                st.markdown(f"**{len(pdf_files)} PDF(s) geladen:**")
                for i, pdf in enumerate(pdf_files, 1):
                    # Truncate long filenames for better display
                    display_name = pdf if len(pdf) <= 40 else pdf[:37] + "..."
                    st.markdown(f"**{i}.** {display_name}")
            else:
                st.warning("Keine PDF-Dateien im data-Ordner gefunden")

        st.markdown("---")
        st.markdown("### ⚙️ System-Status")
        st.success("🟢 System Online")
        st.info("🔑 Google Gemini API Verbunden")

    # Footer with better styling
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; opacity: 0.7; font-size: 0.9rem;'>"
        "🚀 <strong>Angetrieben von</strong> Google Gemini • ChromaDB • Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
