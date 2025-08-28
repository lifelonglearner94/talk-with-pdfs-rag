"""Lightweight metrics helpers used by the API.

Provides in-process counters and simple prometheus exposition when
`prometheus_client` is available. This is intentionally minimal to avoid
hard dependencies in tests.
"""
from collections import Counter, defaultdict
import threading
from typing import Optional

_lock = threading.Lock()
_counters = Counter()
_latencies = defaultdict(list)


def increment_request(endpoint: str) -> None:
    with _lock:
        _counters[f"requests.{endpoint}.count"] += 1


def observe_latency(endpoint: str, seconds: float) -> None:
    with _lock:
        _latencies[endpoint].append(float(seconds))


def generate_prometheus() -> Optional[str]:
    try:
        from prometheus_client import CollectorRegistry, Gauge, generate_latest
    except Exception:
        return None
    reg = CollectorRegistry()
    # expose counters
    for k, v in list(_counters.items()):
        g = Gauge(k.replace('.', '_'), 'auto-generated', registry=reg)
        g.set(v)
    # expose avg latency per endpoint
    for ep, vals in list(_latencies.items()):
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        g = Gauge(f"latency_{ep}_avg_seconds", 'avg latency', registry=reg)
        g.set(avg)
    return generate_latest(reg).decode('utf-8')


def generate_prometheus_from_logs(log_file_path) -> Optional[str]:
    # Best-effort fallback that inspects query_log.jsonl to build a tiny prometheus text
    try:
        import json
        from io import StringIO
        lines = log_file_path.read_text(encoding='utf-8').splitlines()
        total = 0
        lat_sum = 0.0
        lat_count = 0
        for L in lines:
            if not L.strip():
                continue
            try:
                r = json.loads(L)
            except Exception:
                continue
            total += 1
            l = r.get('latency_sec')
            if isinstance(l, (int, float)):
                lat_sum += float(l)
                lat_count += 1
        avg = (lat_sum / lat_count) if lat_count else 0.0
        # Build simple Prometheus exposition
        out = StringIO()
        out.write(f"# HELP talk_with_pdfs_total_queries Total queries recorded\n")
        out.write(f"# TYPE talk_with_pdfs_total_queries counter\n")
        out.write(f"talk_with_pdfs_total_queries {total}\n")
        out.write(f"# HELP talk_with_pdfs_avg_latency_seconds Average query latency seconds\n")
        out.write(f"# TYPE talk_with_pdfs_avg_latency_seconds gauge\n")
        out.write(f"talk_with_pdfs_avg_latency_seconds {avg}\n")
        return out.getvalue()
    except Exception:
        return None


def snapshot():
    """Return a JSON-serializable snapshot of in-memory counters and latency summaries."""
    with _lock:
        counters = dict(_counters)
        lat = {k: {'count': len(v), 'avg': (sum(v)/len(v) if v else 0.0)} for k, v in _latencies.items()}
    return {'counters': counters, 'latencies': lat}
