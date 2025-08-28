"""Safe adapter for the RAG pipeline used by the API.

This module tries to import the project's RAGPipeline and expose a
singleton `get_pipeline()` that returns an object implementing `answer(question)`
and `list_sources()` used by the API. If the real pipeline is unavailable
or fails to initialize, we return a small stub that provides deterministic
responses so tests can run without heavy dependencies.
"""
from typing import List
import json
from pathlib import Path


class _StubPipeline:
    def answer(self, question: str):
        # mimic AnswerResult shape with simple fields used by API tests
        return type("R", (), {"answer": "stub answer", "sources": []})()

    def list_sources(self) -> List[str]:
        return []


_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        # Try importing the project's RAGPipeline
        from app.core.rag_pipeline import RAGPipeline

        try:
            _pipeline = RAGPipeline()
        except Exception:
            # Fail-safe: fall back to stub if pipeline init fails
            _pipeline = _StubPipeline()
    except Exception:
        _pipeline = _StubPipeline()
    return _pipeline


def set_pipeline(pipeline) -> None:
    """Replace the global pipeline singleton (useful for tests).

    Pass an object implementing `answer(question)` and optional
    `list_sources()` to inject a fake or test double pipeline for API tests.
    """
    global _pipeline
    _pipeline = pipeline


def set_pipeline_from_config(overrides: dict | None = None) -> bool:
    """Attempt to create and set a real `RAGPipeline` using a `RAGConfig`.

    Parameters
    - overrides: dict of RAGConfig fields to override (e.g. {'persist_dir': Path('tmp')})

    Returns True if the pipeline was created and set, False otherwise.
    """
    global _pipeline
    try:
        from app.core.config import RAGConfig
        from app.core.rag_pipeline import RAGPipeline
    except Exception:
        return False

    try:
        cfg = RAGConfig.from_env()
        if overrides:
            for k, v in dict(overrides).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        _pipeline = RAGPipeline(cfg)
        return True
    except Exception:
        # don't clobber any existing pipeline on failure
        return False


def reset_pipeline() -> None:
    """Reset the singleton pipeline to force lazy reinitialization.

    Useful in tests to clean up after injection.
    """
    global _pipeline
    _pipeline = None


def list_sources_from_disk() -> List[str]:
    """Try to read a persisted sources file under `vectorstore/`.

    Returns list of source identifiers if found, otherwise empty list.
    """
    base = Path("vectorstore")
    # Common candidate filenames we might persist during indexing
    candidates = [base / "sources.json", base / "index_state.json", base / "sources_list.json"]
    for p in candidates:
        if not p.exists():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # If explicit sources file
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return obj
        # If index_state contains a 'sources' field
        if isinstance(obj, dict) and "sources" in obj and isinstance(obj["sources"], list):
            return [s for s in obj["sources"] if isinstance(s, str)]
    return []
