from pathlib import Path
from app.core import RAGConfig, RAGPipeline
from langchain_core.documents import Document
import tempfile, shutil
from pypdf import PdfWriter

class DummyEmbeddings:
    def embed_documents(self, texts, *_, **__):
        return [[0.0] * 5 for _ in texts]
    def embed_query(self, text, *_, **__):  # pragma: no cover trivial
        return [0.0] * 5

# monkeypatch provider to avoid external API


def test_list_sources_unique_sorted(monkeypatch):
    tmp_dir = Path(tempfile.mkdtemp())
    data_dir = tmp_dir / 'data'
    persist = tmp_dir / 'vec'
    data_dir.mkdir()
    for name in ['z_file.pdf', 'a_file.pdf']:
        w = PdfWriter(); w.add_blank_page(width=200, height=200)
        with (data_dir / name).open('wb') as f:
            w.write(f)

    cfg = RAGConfig(data_dir=data_dir, persist_dir=persist, top_k=2)

    # Replace EmbeddingProvider with dummy
    monkeypatch.setattr('app.core.embeddings.EmbeddingProvider', lambda model: type('X', (), {'embeddings': DummyEmbeddings()})())

    pipe = RAGPipeline(cfg)
    # Monkeypatch splitter to ensure at least one chunk per source (blank PDFs produce empty text)
    original_splitter = pipe.splitter
    def _fake_split(docs):
        out = []
        for d in docs:
            # Guarantee one small chunk per doc
            out.append(Document(page_content="dummy text", metadata=d.metadata))
        return out
    original_splitter.split = _fake_split  # type: ignore
    pipe.ensure_index()

    sources = pipe.vs_manager.list_sources()
    assert sources == sorted(sources)
    assert len(sources) == 2

    shutil.rmtree(tmp_dir, ignore_errors=True)
