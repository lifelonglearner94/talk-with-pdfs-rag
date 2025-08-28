from __future__ import annotations
import logging, time, functools, os

LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO").upper()

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
