import json
from pathlib import Path

from fastapi.testclient import TestClient

from interfaces.api import app


def test_sources_reads_from_vectorstore_disk(tmp_path, monkeypatch):
    # Create a temporary workspace vectorstore directory
    vv = tmp_path / "vectorstore"
    vv.mkdir()
    sources = ["paperA", "paperB", "paperC"]
    (vv / "sources.json").write_text(json.dumps(sources), encoding="utf-8")

    # Monkeypatch cwd to tmp_path so API reads the file
    monkeypatch.chdir(tmp_path)

    client = TestClient(app)
    r = client.get("/sources")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert set(sources) == set(data)
