from __future__ import annotations
import logging, time, functools, os, json

LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO").upper()

if os.getenv("RAG_LOG_JSON", "0") == "1":
    class _JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # pragma: no cover simple json
            payload = {
                "ts": int(record.created * 1000),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            return json.dumps(payload, ensure_ascii=False)
    formatter = _JsonFormatter()
else:
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
_handler = logging.StreamHandler()
_handler.setFormatter(formatter)

logger = logging.getLogger("rag")
if not logger.handlers:
    logger.addHandler(_handler)
logger.setLevel(LOG_LEVEL)


def timed(name: str | None = None):
    def deco(func):
        label = name or func.__name__
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                dur = (time.perf_counter() - start) * 1000
                logger.debug(f"timing ms={dur:.1f} step={label}")
        return wrapper
    return deco
