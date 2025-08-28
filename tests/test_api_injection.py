from fastapi.testclient import TestClient

from interfaces.api import app
from interfaces import pipeline_adapter


class FakePipeline:
    def __init__(self, answer_text="injected answer", sources=None):
        self._answer = answer_text
        self._sources = sources or []

    def answer(self, question: str):
        return type("R", (), {"answer": self._answer, "sources": self._sources})()

    def list_sources(self):
        return ["paper1", "paper2"]


def test_ask_uses_injected_pipeline():
    fake = FakePipeline(answer_text="injected answer", sources=["s1"])
    pipeline_adapter.set_pipeline(fake)
    client = TestClient(app)
    r = client.post("/ask", json={"question": "Why?"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "injected answer"
    # cleanup
    pipeline_adapter.reset_pipeline()


def test_sources_uses_injected_pipeline():
    fake = FakePipeline()
    pipeline_adapter.set_pipeline(fake)
    client = TestClient(app)
    r = client.get("/sources")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and "paper1" in data
    pipeline_adapter.reset_pipeline()
