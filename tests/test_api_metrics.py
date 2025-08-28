from fastapi.testclient import TestClient
from pathlib import Path
import json

from interfaces.api import app


def write_log(lines):
    p = Path("logs")
    p.mkdir(exist_ok=True)
    (p / "query_log.jsonl").write_text("\n".join(json.dumps(l) for l in lines), encoding="utf-8")


def test_metrics_summary_json(tmp_path, monkeypatch):
    # Prepare a small log
    lines = [
        {"question": "q1", "latency_sec": 0.12, "retrieval_mode": "vector", "sources": ["s1"]},
        {"question": "q2", "latency_sec": 0.24, "retrieval_mode": "keyword", "sources": ["s2", "s1"]},
    ]
    # monkeypatch current working dir to tmp_path so logs/ is isolated
    monkeypatch.chdir(tmp_path)
    write_log(lines)
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["total_queries"] == 2
    assert abs(data["avg_latency_sec"] - 0.18) < 1e-6
    assert "vector" in data["retrieval_mode_counts"]
    assert "s1" in data["top_sources"]
