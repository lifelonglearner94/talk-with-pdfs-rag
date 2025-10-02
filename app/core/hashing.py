from __future__ import annotations
from pathlib import Path
import json
import hashlib
from typing import Sequence
from .config import RAGConfig

INDEX_FORMAT_VERSION = 2  # bumped for page range + deterministic chunk_id schema changes


def compute_index_hash(pdf_paths: Sequence[Path], config: RAGConfig) -> str:
    items = []
    for p in sorted(pdf_paths):
        if not p.exists():
            continue
        stat = p.stat()
        items.append({
            "name": p.name,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        })
    payload = {
        "files": items,
        "params": config.hash_relevant_params(),
        "version": INDEX_FORMAT_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.md5(raw).hexdigest()


def load_index_state(persist_dir: Path) -> dict | None:
    """Load the stored index state (hash + optional version).

    Backward compatible: older files may only contain {"hash": "..."}.
    """
    f = persist_dir / "index_state.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
        return data
    except Exception:
        return None


def load_stored_hash(persist_dir: Path) -> str | None:  # backward compatibility helper
    state = load_index_state(persist_dir)
    return state.get("hash") if state else None


def store_hash(persist_dir: Path, the_hash: str):
    f = persist_dir / "index_state.json"
    payload = json.dumps({"hash": the_hash, "version": INDEX_FORMAT_VERSION}, indent=2)
    # Atomic write: write to temp file then replace
    try:
        tmp = persist_dir / "index_state.json.tmp"
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(f)
    except Exception:
        # Best-effort: fall back to direct write if replace fails
        try:
            f.write_text(payload, encoding="utf-8")
        except Exception:
            # If even that fails, there's nothing more to do here
            pass
