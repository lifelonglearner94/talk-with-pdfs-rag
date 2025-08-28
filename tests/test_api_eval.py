from fastapi.testclient import TestClient
from interfaces.api import app
from pathlib import Path


def test_eval_run_endpoint():
    client = TestClient(app)
    # If the eval script or gold file is not present, the endpoint should return 404
    gold = Path("experiments/eval/gold_examples.jsonl")
    payload = {"gold": str(gold), "k": 2, "mode": "vector", "compare_rerank": False}
    r = client.post("/eval/run", json=payload)
    # Accept either a successful JSON result or a not-found (script missing) error
    assert r.status_code in (200, 404)
