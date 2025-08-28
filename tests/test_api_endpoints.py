from fastapi.testclient import TestClient

from interfaces.api import app


def test_ask_endpoint():
    client = TestClient(app)
    r = client.post("/ask", json={"question": "What is the purpose of this API?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data and isinstance(data["answer"], str)
    assert "citations" in data and isinstance(data["citations"], list)


def test_sources_endpoint():
    client = TestClient(app)
    r = client.get("/sources")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
