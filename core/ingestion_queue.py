"""Async ingestion queue scaffold.

Provides a lightweight asyncio-based queue for batching document embeddings and
calling an injectable worker to persist to the vector backend.
"""
import asyncio
from typing import Callable, Iterable, Mapping, Any, List, Optional
import sqlite3
from pathlib import Path


class IngestionQueue:
    def __init__(self, worker: Callable[[List[Mapping[str, Any]]], None], batch_size: int = 16, interval: float = 1.0):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker = worker
        self._batch_size = batch_size
        self._interval = interval
        self._task: Optional[asyncio.Task] = None
        self._stopped = False

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._stopped = True
        if self._task:
            await self._task

    async def push(self, item: Mapping[str, Any]):
        await self._queue.put(item)

    async def _run(self):
        buffer: List[Mapping[str, Any]] = []
        while not self._stopped:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self._interval)
                buffer.append(item)
            except asyncio.TimeoutError:
                pass

            if len(buffer) >= self._batch_size or (buffer and self._queue.empty()):
                try:
                    await self._maybe_await_worker(buffer)
                except Exception:
                    # Worker errors shouldn't kill the queue; log in real impl
                    pass
                buffer = []

    async def _maybe_await_worker(self, batch: List[Mapping[str, Any]]):
        result = self._worker(batch)
        if asyncio.iscoroutine(result):
            await result


def worker_from_backend(backend, index_name: str):
    """Return a worker callable that will create the index (once) and add documents in batches.

    The returned worker accepts a list of dicts where each dict must contain at least
    'id', 'text', 'metadata', and optionally 'embedding'. This helper keeps a small
    in-memory flag to avoid recreating the index multiple times.
    """
    created = False
    # Dedupe: prefer on-disk sqlite-backed store for durability across restarts.
    import hashlib
    # Prefer a backend-provided persist directory to keep dedupe DB colocated.
    # Use a per-index filename (includes index_name) so different indices or
    # test fixtures don't interfere with each other's dedupe state.
    bp = getattr(backend, "persist_directory", None)
    safe_name = index_name.replace('/', '_').replace('..', '_') if index_name else 'default'
    db_name = f".ingestion_dedupe_{safe_name}.sqlite"
    if bp:
        dedupe_db = Path(bp) / db_name
    else:
        dedupe_db = Path(db_name)
    conn: Optional[sqlite3.Connection] = None

    def _ensure_db():
        nonlocal conn
        if conn is not None:
            return
        dedupe_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(dedupe_db))
        conn.execute("""CREATE TABLE IF NOT EXISTS seen_fingerprints(
            fp TEXT PRIMARY KEY,
            created_at INTEGER
        )""")
        conn.commit()

    def _db_has(fp: str) -> bool:
        try:
            _ensure_db()
            cur = conn.execute("SELECT 1 FROM seen_fingerprints WHERE fp=?", (fp,))
            return cur.fetchone() is not None
        except Exception:
            # On any DB error, fall back to allowing the doc through
            return False

    def _db_add(fp: str) -> None:
        try:
            _ensure_db()
            conn.execute("INSERT OR IGNORE INTO seen_fingerprints(fp, created_at) VALUES (?, strftime('%s','now'))", (fp,))
            conn.commit()
        except Exception:
            pass

    def _doc_fingerprint(d: Mapping[str, Any]) -> str:
        if d.get("id"):
            return f"id:{d.get('id')}"
        text = (d.get("text") or "")
        cit = ""
        md = d.get("metadata") or {}
        if isinstance(md, dict):
            cit = md.get("citation_key") or md.get("source") or ""
        h = hashlib.md5((text + str(cit)).encode("utf-8", errors="ignore")).hexdigest()
        return f"txt:{h}"

    def _worker(docs: List[Mapping[str, Any]]):
        nonlocal created
        if not created:
            backend.create_index(index_name)
            created = True
        # Filter duplicates using on-disk DB with in-memory fallback
        new_docs = []
        for d in docs:
            fp = _doc_fingerprint(d)
            seen_before = _db_has(fp)
            if seen_before:
                continue
            _db_add(fp)
            new_docs.append(d)
        if not new_docs:
            return
        # backend expects a sequence of mappings
        backend.add_documents(index_name, new_docs)

    return _worker
