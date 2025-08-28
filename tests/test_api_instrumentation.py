from fastapi.testclient import TestClient

from interfaces.api import app


client = TestClient(app)


def test_instrumentation_snapshot():
    # Hit a couple endpoints to generate counters and latencies
    r1 = client.get("/health")
    assert r1.status_code == 200
    r2 = client.get("/sources")
    assert r2.status_code == 200

    # Request metrics snapshot
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "in_memory_counters" in data
    # We expect at least 'health' and 'sources' counters
    counters = data["in_memory_counters"]
    assert any(k.endswith('health') or k.endswith('sources') or k == 'requests.health.count' for k in counters.keys())
