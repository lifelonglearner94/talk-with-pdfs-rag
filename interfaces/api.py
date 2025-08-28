"""Minimal FastAPI service scaffold for Phase 3.

Endpoints:
- /health
- /ask (POST) - simple passthrough to existing pipeline (stub)
- /sources (GET) - list available source ids (stub)

This module is intentionally light-weight and uses dependency injection
so it can be wired into the existing codebase later.
"""
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from collections import Counter
import json
from pathlib import Path

from .pipeline_adapter import get_pipeline
from . import metrics as metrics_module
from . import pipeline_adapter
from pathlib import Path as _Path
from subprocess import Popen, PIPE
from fastapi import Request
import time as _time


app = FastAPI(title="talk_with_pdfs API", version="0.1.0")

# Enable permissive CORS for local development; production should tighten this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    # record per-endpoint request count and latency
    path = request.url.path.strip('/') or 'root'
    metrics_module.increment_request(path)
    t0 = _time.time()
    resp = await call_next(request)
    latency = _time.time() - t0
    metrics_module.observe_latency(path, latency)
    return resp


# Lightweight per-client token-bucket rate limiter (very small default limits)
try:
    from .rate_limiter import RateLimiter, TooManyRequests
    _rate_limiter = RateLimiter(rate=5, capacity=10)

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        client = request.client.host if request.client else "anon"
        try:
            _rate_limiter.consume(client)
        except TooManyRequests:
            raise HTTPException(status_code=429, detail="too many requests")
        return await call_next(request)
except Exception:
    # If rate limiter cannot be imported, skip adding the middleware
    _rate_limiter = None


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class AskResponse(BaseModel):
    answer: str
    citations: List[str] = []


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="question required")
    metrics_module.increment_request("ask")
    pipeline = get_pipeline()
    try:
        # Support both sync and async pipeline.answer implementations
        import time
        t0 = time.time()
        maybe = pipeline.answer(req.question)
        if hasattr(maybe, "__await__"):
            # coroutine - await it
            result = await maybe
        else:
            result = maybe
        latency = time.time() - t0
        metrics_module.observe_latency("ask", latency)
    except Exception:
        # Fall back to a stub response if pipeline fails
        raise HTTPException(status_code=500, detail="pipeline error")
    # Normalize result: support both AnswerResult-like objects and simple stubs
    answer_text = getattr(result, "answer", str(result))
    sources = getattr(result, "sources", []) or []
    citations = [s.chunk_id if hasattr(s, "chunk_id") else s for s in sources]
    return AskResponse(answer=answer_text, citations=citations)


@app.get("/sources", response_model=List[str])
async def sources():
    # Prefer an injected or initialized pipeline's listing when available.
    pipeline = get_pipeline()
    try:
        if hasattr(pipeline, "list_sources"):
            res = pipeline.list_sources()
            if res:
                return res
        # Try to access vs_manager if RAGPipeline
        if hasattr(pipeline, "vs_manager") and getattr(pipeline.vs_manager, "list_sources", None):
            res2 = pipeline.vs_manager.list_sources()
            if res2:
                return res2
    except Exception:
        # If pipeline listing fails, fall back to disk-based listing
        pass

    # Fall back to any persisted sources on disk (index metadata)
    try:
        from .pipeline_adapter import list_sources_from_disk

        disk_sources = list_sources_from_disk()
        if disk_sources:
            return disk_sources
    except Exception:
        pass

    return []


@app.get("/metrics")
async def metrics(format: Optional[str] = None):
    """Return simple query-log derived metrics as JSON, or Prometheus exposition if
    `format=prometheus` and `prometheus_client` is installed.
    """
    log_file = Path("logs") / "query_log.jsonl"
    if format == "prometheus":
        # Prefer server-side Prometheus exposition if available
        prom = metrics_module.generate_prometheus()
        if prom is not None:
            return Response(content=prom, media_type="text/plain; version=0.0.4")

    if not log_file.exists():
        return {"total_queries": 0, "avg_latency_sec": None, "retrieval_mode_counts": {}, "top_sources": []}
    records = []
    try:
        for line in log_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        raise HTTPException(status_code=500, detail="failed to read query log")

    total = len(records)
    latencies = [r.get("latency_sec") for r in records if isinstance(r.get("latency_sec"), (int, float))]
    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else None
    modes = Counter(r.get("retrieval_mode") for r in records if r.get("retrieval_mode"))
    # Flatten sources and count
    srcs = Counter()
    for r in records:
        s = r.get("sources") or []
        if isinstance(s, list):
            for x in s:
                if x:
                    srcs[x] += 1
    top_sources = [s for s, _ in srcs.most_common(10)]

    # Merge with in-memory metrics snapshot
    snap = metrics_module.snapshot()
    result = {
        "total_queries": total,
        "avg_latency_sec": avg_latency,
        "retrieval_mode_counts": dict(modes),
        "top_sources": top_sources,
        "in_memory_counters": snap.get('counters', {}),
        "in_memory_latencies": snap.get('latencies', {}),
    }

    if format == "prometheus":
        # Older fallback: try to generate from the logs via metrics module
        prom = metrics_module.generate_prometheus_from_logs(log_file)
        if prom is not None:
            return Response(content=prom, media_type="text/plain; version=0.0.4")

    return result


class AdminInitRequest(BaseModel):
    overrides: Optional[dict] = None


@app.post("/admin/init_pipeline")
async def admin_init_pipeline(req: AdminInitRequest):
    """Attempt to initialize and set a real RAGPipeline from environment/config overrides.

    This endpoint is intended for local development and testing so tests or
    a developer can deterministically initialize the pipeline used by `/ask`.
    """
    ok = pipeline_adapter.set_pipeline_from_config(req.overrides or {})
    if not ok:
        raise HTTPException(status_code=500, detail="pipeline initialization failed")
    return {"status": "ok", "message": "pipeline initialized"}


@app.post("/admin/reset_pipeline")
async def admin_reset_pipeline():
    """Reset the in-process pipeline singleton so subsequent calls lazily recreate it."""
    pipeline_adapter.reset_pipeline()
    return {"status": "ok", "message": "pipeline reset"}


@app.post("/admin/ensure_index")
async def admin_ensure_index(force: Optional[bool] = False):
    """Trigger `RAGPipeline.ensure_index(force=True)` if a real pipeline is available.

    Returns 404 if no real pipeline implementation is present.
    """
    pipeline = get_pipeline()
    # If the pipeline is the stub, refuse to run an expensive ensure_index
    if getattr(pipeline, "__class__", None) and pipeline.__class__.__name__ == "_StubPipeline":
        raise HTTPException(status_code=404, detail="real pipeline not initialized")
    try:
        # allow sync or async ensure_index
        maybe = pipeline.ensure_index(force=force)
        if hasattr(maybe, "__await__"):
            await maybe
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ensure_index failed: {e}")
    return {"status": "ok", "message": "ensure_index completed"}


class EvalRequest(BaseModel):
    gold: Optional[str] = None
    k: Optional[int] = 10
    mode: Optional[str] = "vector"
    compare_rerank: Optional[bool] = False


@app.post("/eval/run")
async def run_eval_endpoint(req: EvalRequest):
    """Run the evaluation harness (safe, focused run).

    This endpoint is intended for local/dev use. It will invoke the
    `experiments/eval/run_eval.py` script with provided parameters. The
    implementation is defensive: it returns an error if the script is
    missing or if the requested gold file does not exist.
    """
    script = _Path("experiments/eval/run_eval.py")
    if not script.exists():
        raise HTTPException(status_code=404, detail="evaluation script not found")
    gold_path = req.gold or "experiments/eval/gold_examples.jsonl"
    gp = _Path(gold_path)
    if not gp.exists():
        raise HTTPException(status_code=404, detail=f"gold file not found: {gold_path}")
    # Build command (use the same python env as the running process)
    cmd = ["python", str(script), "--gold", str(gp), "--k", str(req.k or 10), "--mode", req.mode or "vector"]
    if req.compare_rerank:
        cmd.append("--compare-rerank")
    # Run the script but limit runtime in case of issues; capture output
    try:
        p = Popen(cmd, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate(timeout=60)
    except TimeoutError as e:
        try:
            p.kill()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"evaluation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"evaluation failed: {e}")
    if p.returncode != 0:
        raise HTTPException(status_code=500, detail=f"evaluation script error: {err.decode('utf-8', errors='ignore')}")
    try:
        # The script writes results to experiments/eval/results; parse stdout for path
        text = out.decode("utf-8", errors="ignore")
        # Best-effort: return the JSON file content if present
        res_dir = _Path("experiments/eval/results")
        if res_dir.exists():
            latest = sorted(res_dir.glob("*.json"))[-1]
            return json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"status": "ok", "message": "evaluation completed; check experiments/eval/results"}


@app.get("/eval/status")
async def eval_status():
    """Return metadata for the most recent evaluation run if available.

    Looks for JSON files under `experiments/eval/results` and returns the latest one
    parsed. If none exist, returns a 404-like empty response.
    """
    res_dir = Path("experiments/eval/results")
    if not res_dir.exists():
        return {"last_eval": None}
    files = sorted(res_dir.glob("*.json"))
    if not files:
        return {"last_eval": None}
    latest = files[-1]
    try:
        obj = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="failed to read latest eval result")
    # Return compact metadata only to avoid huge payloads
    meta = {
        "path": str(latest),
        "summary": obj.get("summary"),
        "elapsed_sec": obj.get("elapsed_sec"),
        "k": obj.get("k"),
        "mode": obj.get("mode"),
        "rerank_compared": obj.get("rerank_compared", False),
    }
    return {"last_eval": meta}
