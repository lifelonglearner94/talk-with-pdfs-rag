from fastapi.testclient import TestClient

from interfaces.api import app
from interfaces import pipeline_adapter


class AsyncFakePipeline:
    def __init__(self, answer_text="async injected", sources=None):
        self._answer = answer_text
        self._sources = sources or []

    async def answer(self, question: str):
        # simulate an async operation
        return type("R", (), {"answer": self._answer, "sources": self._sources})()


def test_async_pipeline_answer():
    fake = AsyncFakePipeline(answer_text="async injected", sources=["s_async"])
    pipeline_adapter.set_pipeline(fake)
    client = TestClient(app)
    r = client.post("/ask", json={"question": "How?"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "async injected"
    assert "s_async" in data.get("citations", [])
    pipeline_adapter.reset_pipeline()
