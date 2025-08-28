from fastapi.testclient import TestClient

from interfaces.api import app


client = TestClient(app)


def test_metrics_json():
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "total_queries" in data and "avg_latency_sec" in data


def test_metrics_prometheus_fallback():
    # Request prometheus format; if prometheus_client isn't installed we accept JSON fallback
    r = client.get("/metrics?format=prometheus")
    assert r.status_code == 200
    # If text/plain returned, check content-type; otherwise accept JSON
    if "text/plain" in r.headers.get("content-type", ""):
        assert r.text.startswith("#") or "talk_with_pdfs_total_queries" in r.text
    else:
        data = r.json()
        assert "total_queries" in data
