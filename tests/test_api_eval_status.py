import json
from pathlib import Path
from fastapi.testclient import TestClient
from interfaces.api import app


def test_eval_status_returns_latest(tmp_path, monkeypatch):
    # Prepare a fake results dir and a fake results JSON
    res_dir = Path('experiments/eval/results')
    res_dir.mkdir(parents=True, exist_ok=True)
    fake = {
        "summary": {"retrieval_recall@k": 0.5},
        "elapsed_sec": 1.23,
        "k": 2,
        "mode": "vector",
    }
    # write file
    p = res_dir / "eval_test_fake.json"
    p.write_text(json.dumps(fake), encoding='utf-8')

    client = TestClient(app)
    r = client.get('/eval/status')
    assert r.status_code == 200
    data = r.json()
    assert data.get('last_eval') is not None
    assert data['last_eval']['k'] == 2
