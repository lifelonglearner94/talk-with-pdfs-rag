from __future__ import annotations
from pathlib import Path
import json
import hashlib
from typing import Sequence
from .config import RAGConfig

INDEX_FORMAT_VERSION = 1


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


def load_stored_hash(persist_dir: Path) -> str | None:
    f = persist_dir / "index_state.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
        return data.get("hash")
    except Exception:
        return None


def store_hash(persist_dir: Path, the_hash: str):
    f = persist_dir / "index_state.json"
    f.write_text(json.dumps({"hash": the_hash}, indent=2))
