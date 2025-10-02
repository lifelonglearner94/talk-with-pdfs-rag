# Talk with PDFs

RAG pipeline for chatting with scientific PDFs using LangChain, Chroma, and Google Gemini.

## Features

- **Bibliography Removal**: Auto-detects and removes reference sections (EN/DE)
- **Smart Retrieval**: Vector, BM25, and hybrid search with optional reranking
- **Section-Aware Chunking**: Preserves document structure and context
- **Citation Tracking**: Automatic source attribution

## Quick Start

1. **Setup**
   ```bash
   # Install dependencies
   uv sync

   # Configure API key
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

2. **Add PDFs** to the `data/` folder

3. **Run**
   ```bash
   # Streamlit UI
   uv run streamlit run app/interfaces/streamlit_app.py

   # Or CLI
   uv run rag-pdf-chat
   ```

## Docker

```bash
docker build -t talk-with-pdfs .
docker run -p 8501:8501 -v $(pwd)/data:/app/data -e GOOGLE_API_KEY=your_key talk-with-pdfs
```

## Configuration

- `GOOGLE_API_KEY`: Required for Google Gemini
- `RAG_REMOVE_BIBLIOGRAPHY`: Enable/disable bibliography removal (default: `true`)

## License

MIT
