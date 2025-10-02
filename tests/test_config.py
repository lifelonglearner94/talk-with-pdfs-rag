from app.core import RAGConfig

def test_config_override_env(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "7")
    cfg = RAGConfig.from_env()
    assert cfg.top_k == 7
