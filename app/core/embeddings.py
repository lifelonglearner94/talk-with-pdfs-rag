from __future__ import annotations
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio, time, random, os
from .logging import logger


class _RetryingEmbeddings:
    """Thin wrapper adding exponential backoff to an embeddings client.

    Retries on generic Exceptions whose message or type indicates a transient
    rate limit / quota / 429 style error. Keeps logic simple and dependency-light.
    """

    RETRIABLE_KEYWORDS = [
        "rate", "quota", "429", "exceed", "exceeded", "temporarily", "ResourceExhausted"
    ]

    def __init__(
        self,
        inner,
        max_retries=6,
        base_delay=0.5,
        backoff_factor=2.0,
        max_delay=30.0,
        jitter=True,
        batch_size: int = 32,
        requests_per_min: int | None = None,
        long_quota_pause: float = 60.0,
    ):
        self._inner = inner
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.jitter = jitter
        self.batch_size = max(1, batch_size)
        self.requests_per_min = requests_per_min
        self._min_interval = 60.0 / requests_per_min if requests_per_min else 0.0
        self._last_call_ts: float | None = None
        self.long_quota_pause = long_quota_pause

    # ------------- internal helpers -------------
    def _should_retry(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(k in msg for k in self.RETRIABLE_KEYWORDS)

    def _sleep(self, attempt: int):
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        time.sleep(delay)

    def _run_with_retries(self, fn_name: str, *args, **kwargs):
        attempt = 0
        while True:
            try:
                fn = getattr(self._inner, fn_name)
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                if attempt >= self.max_retries or not self._should_retry(e):
                    logger.warning(
                        "embeddings.%s failed (attempt=%d) giving up: %s", fn_name, attempt, e
                    )
                    raise
                quota_like = any(k in str(e).lower() for k in ["quota", "per minute", "exceeded", "resourceexhausted"])
                if quota_like and attempt == 0:
                    logger.warning(
                        "embeddings.%s quota warning detected -> applying long pause %.1fs before retries", fn_name, self.long_quota_pause
                    )
                    time.sleep(self.long_quota_pause)
                logger.warning(
                    "embeddings.%s transient error attempt=%d/%d: %s -> backoff", fn_name, attempt + 1, self.max_retries, e
                )
                self._sleep(attempt)
                attempt += 1

    # ------------- public API (subset) -------------
    def _throttle(self):
        if self._min_interval <= 0:
            return
        now = time.time()
        if self._last_call_ts is None:
            self._last_call_ts = now
            return
        elapsed = now - self._last_call_ts
        if elapsed < self._min_interval:
            sleep_for = self._min_interval - elapsed
            time.sleep(sleep_for)
        self._last_call_ts = time.time()

    def embed_documents(self, texts, *args, **kwargs):  # langchain core uses this
        # Batch + throttle to spread out requests across the minute window.
        if not texts:
            return []
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            self._throttle()
            logger.debug(
                "embeddings.batch start idx=%d size=%d total=%d", i, len(batch), len(texts)
            )
            embs = self._run_with_retries("embed_documents", batch, *args, **kwargs)
            all_embs.extend(embs)
        return all_embs

    def embed_query(self, text, *args, **kwargs):
        return self._run_with_retries("embed_query", text, *args, **kwargs)

    # pass-through for any other attributes (e.g., client configs)
    def __getattr__(self, item):  # pragma: no cover simple passthrough
        return getattr(self._inner, item)

class EmbeddingProvider:
    def __init__(self, model: str):
        self.model_name = model
        # Streamlit executes the script in a background thread that has no default
        # asyncio event loop. google-generative-ai (grpc.aio) needs a loop when the
        # async client is created. Ensure one exists for this thread.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop in this thread – create and set one.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        raw = GoogleGenerativeAIEmbeddings(model=model)
        # Backoff tuning via env (optional)
        max_retries = int(os.getenv("RAG_EMBED_MAX_RETRIES", "6"))
        base_delay = float(os.getenv("RAG_EMBED_BASE_DELAY", "0.5"))
        backoff_factor = float(os.getenv("RAG_EMBED_BACKOFF_FACTOR", "2.0"))
        max_delay = float(os.getenv("RAG_EMBED_MAX_DELAY", "30"))
        batch_size = int(os.getenv("RAG_EMBED_BATCH_SIZE", "32"))
        rpm_env = os.getenv("RAG_EMBED_REQUESTS_PER_MIN")
        rpm = int(rpm_env) if rpm_env else None
        long_quota_pause = float(os.getenv("RAG_EMBED_LONG_QUOTA_PAUSE", "60"))
        self._emb = _RetryingEmbeddings(
            raw,
            max_retries=max_retries,
            base_delay=base_delay,
            backoff_factor=backoff_factor,
            max_delay=max_delay,
            jitter=True,
            batch_size=batch_size,
            requests_per_min=rpm,
            long_quota_pause=long_quota_pause,
        )

    @property
    def embeddings(self):
        return self._emb
