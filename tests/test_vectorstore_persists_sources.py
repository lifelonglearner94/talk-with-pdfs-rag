import json
from pathlib import Path

from langchain_core.documents import Document


def _fake_from_documents(documents, embedding, persist_directory):
    # return a minimal fake object with expected interface used elsewhere
    class FakeVS:
        def __init__(self):
            self._metas = [d.metadata or {} for d in documents]

        def get(self, include=None, where=None, limit=None):
            return {"metadatas": self._metas}

    return FakeVS()


def test_build_persists_sources(tmp_path, monkeypatch):
    vs_dir = tmp_path / "vectorstore"
    vs_dir.mkdir()

    # Create two fake documents with metadata
    docs = [
        Document(page_content="a", metadata={"citation_key": "Alpha2024", "source": "a.pdf"}),
        Document(page_content="b", metadata={"source_name": "BetaPaper", "source": "b.pdf"}),
    ]

    # Monkeypatch Chroma.from_documents to avoid real backend
    import app.core.vectorstore as vs_mod

    monkeypatch.setattr(vs_mod, "Chroma", type("C", (), {"from_documents": staticmethod(lambda **kwargs: _fake_from_documents(kwargs.get('documents', []), None, None))}))

    # Create a dummy embedding function
    emb = lambda x: x

    manager = vs_mod.VectorStoreManager(vs_dir, emb)
    # minimal fake config with required method used by compute_index_hash
    class FakeConfig:
        def hash_relevant_params(self):
            return {}

    # call build - should create sources.json
    manager.build(docs, config=FakeConfig())

    sf = vs_dir / "sources.json"
    assert sf.exists()
    data = json.loads(sf.read_text(encoding="utf-8"))
    assert "Alpha2024" in data
    assert "BetaPaper" in data
