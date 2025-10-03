from __future__ import annotations
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio, time, random, os
from .logging import logger


class _RetryingEmbeddings:
    """Thin wrapper adding exponential backoff to an embeddings client.

    Retries on generic Exceptions whose message or type indicates a transient
    rate limit / quota / 429 style error. Keeps logic simple and dependency-light.

    Optimized for Gemini's per-minute quota system (RPM):
    - Tracks when quota errors occur within the 60-second window
    - Waits intelligently until the quota resets (never more than 60s)
    - Ensures eventual success without excessive waiting
    """

    RETRIABLE_KEYWORDS = [
        "rate", "quota", "429", "exceed", "exceeded", "temporarily", "ResourceExhausted"
    ]

    def __init__(
        self,
        inner,
        max_retries=15,  # Enough retries without being excessive
        base_delay=0.5,  # Start with quick retry
        backoff_factor=1.5,  # Gentler exponential growth
        max_delay=70.0,  # Never wait more than 70s (quota resets per minute!)
        jitter=True,
        batch_size: int = 16,  # Smaller batches to avoid hitting limits
        requests_per_min: int | None = None,
        quota_reset_window: float = 70.0,  # Quota resets every 70 seconds (RPM)
    ):
        self._inner = inner
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.max_delay = min(max_delay, quota_reset_window)  # Cap at quota window
        self.jitter = jitter
        self.batch_size = max(1, batch_size)
        self.requests_per_min = requests_per_min
        self._min_interval = 60.0 / requests_per_min if requests_per_min else 0.0
        self._last_call_ts: float | None = None
        self.quota_reset_window = quota_reset_window
        self._quota_error_timestamps: list[float] = []  # Track when quota errors occur

    # ------------- internal helpers -------------
    def _should_retry(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(k in msg for k in self.RETRIABLE_KEYWORDS)

    def _is_quota_error(self, exc: Exception) -> bool:
        """Check if error is specifically a quota/rate limit error."""
        msg = str(exc).lower()
        return any(k in msg for k in ["quota", "per minute", "exceeded", "resourceexhausted", "429"])

    def _sleep(self, attempt: int):
        """Calculate and execute exponential backoff sleep with jitter."""
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        logger.info(
            "embeddings.retry sleeping for %.2f seconds (attempt %d)", delay, attempt + 1
        )
        time.sleep(delay)

    def _smart_quota_pause(self, fn_name: str):
        """Smart quota handling based on per-minute quota window.

        Since Gemini quotas are per minute (RPM), we calculate the optimal wait:
        - Track when quota errors occur
        - Wait until the oldest error is outside the 60s window
        - This ensures we don't wait unnecessarily
        """
        now = time.time()
        self._quota_error_timestamps.append(now)

        # Clean up old timestamps outside the quota window
        cutoff = now - self.quota_reset_window
        self._quota_error_timestamps = [ts for ts in self._quota_error_timestamps if ts > cutoff]

        # Calculate smart wait time
        if self._quota_error_timestamps:
            oldest_error = self._quota_error_timestamps[0]
            time_since_oldest = now - oldest_error

            # If oldest error is recent, wait until it expires from the window
            if time_since_oldest < self.quota_reset_window:
                # Wait until oldest error exits the window, plus a small buffer
                wait_time = (self.quota_reset_window - time_since_oldest) + 2.0

                logger.warning(
                    "embeddings.%s quota exceeded (RPM limit). "
                    "Waiting %.1fs for quota window to reset (%d errors in last %.0fs)",
                    fn_name,
                    wait_time,
                    len(self._quota_error_timestamps),
                    self.quota_reset_window
                )
            else:
                # Oldest error already expired, just brief pause
                wait_time = 2.0
                logger.info(
                    "embeddings.%s brief pause (%.1fs) - quota should be reset",
                    fn_name,
                    wait_time
                )
        else:
            # Fallback: standard wait
            wait_time = min(10.0, self.quota_reset_window / 6)
            logger.warning(
                "embeddings.%s quota error - applying standard pause of %.1fs",
                fn_name,
                wait_time
            )

        # Show countdown for waits > 10 seconds
        if wait_time > 10:
            logger.info("Waiting for quota window to reset...")
            remaining = wait_time
            while remaining > 0:
                sleep_chunk = min(10, remaining)
                time.sleep(sleep_chunk)
                remaining -= sleep_chunk
                if remaining > 5:
                    logger.info("  %.0f seconds remaining...", remaining)
        else:
            time.sleep(wait_time)

    def _run_with_retries(self, fn_name: str, *args, **kwargs):
        """Execute function with robust retry logic optimized for per-minute quotas."""
        attempt = 0
        while True:
            try:
                fn = getattr(self._inner, fn_name)
                result = fn(*args, **kwargs)

                # Success! Clean up old quota error timestamps
                if self._quota_error_timestamps:
                    logger.debug(
                        "embeddings.%s succeeded after quota error(s) - window will reset",
                        fn_name
                    )

                return result

            except Exception as e:  # noqa: BLE001
                # Check if we should retry
                if not self._should_retry(e):
                    logger.error(
                        "embeddings.%s failed with non-retriable error: %s", fn_name, e
                    )
                    raise

                # Check if we've exhausted retries
                if attempt >= self.max_retries:
                    logger.error(
                        "embeddings.%s failed after %d attempts, giving up: %s",
                        fn_name,
                        attempt,
                        e
                    )
                    raise

                # Handle quota errors with smart pause based on RPM window
                if self._is_quota_error(e):
                    logger.warning(
                        "embeddings.%s quota error on attempt %d/%d: %s",
                        fn_name,
                        attempt + 1,
                        self.max_retries,
                        str(e)[:200]  # Truncate long error messages
                    )
                    self._smart_quota_pause(fn_name)
                else:
                    # Non-quota retriable error (e.g., network glitch)
                    logger.warning(
                        "embeddings.%s transient error on attempt %d/%d: %s -> standard backoff",
                        fn_name,
                        attempt + 1,
                        self.max_retries,
                        str(e)[:200]
                    )
                    self._sleep(attempt)

                attempt += 1

    # ------------- public API (subset) -------------
    def _throttle(self):
        """Throttle requests to respect rate limits."""
        if self._min_interval <= 0:
            return
        now = time.time()
        if self._last_call_ts is None:
            self._last_call_ts = now
            return
        elapsed = now - self._last_call_ts
        if elapsed < self._min_interval:
            sleep_for = self._min_interval - elapsed
            logger.debug("embeddings.throttle waiting %.2f seconds", sleep_for)
            time.sleep(sleep_for)
        self._last_call_ts = time.time()

    def embed_documents(self, texts, *args, **kwargs):  # langchain core uses this
        """Batch + throttle to spread out requests across the minute window.

        Enhanced with progress logging for large document sets.
        """
        if not texts:
            return []

        total_texts = len(texts)
        all_embs = []

        logger.info(
            "embeddings.embed_documents processing %d texts in batches of %d",
            total_texts,
            self.batch_size
        )

        for i in range(0, total_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_texts + self.batch_size - 1) // self.batch_size

            self._throttle()

            logger.info(
                "embeddings.batch %d/%d: processing %d texts (progress: %d/%d = %.1f%%)",
                batch_num,
                total_batches,
                len(batch),
                i + len(batch),
                total_texts,
                ((i + len(batch)) / total_texts) * 100
            )

            embs = self._run_with_retries("embed_documents", batch, *args, **kwargs)
            all_embs.extend(embs)

            logger.debug(
                "embeddings.batch %d/%d completed successfully",
                batch_num,
                total_batches
            )

        logger.info(
            "embeddings.embed_documents completed: %d embeddings generated",
            len(all_embs)
        )
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
        # Backoff tuning via env (optional) - optimized for per-minute quotas
        max_retries = int(os.getenv("RAG_EMBED_MAX_RETRIES", "15"))
        base_delay = float(os.getenv("RAG_EMBED_BASE_DELAY", "0.5"))
        backoff_factor = float(os.getenv("RAG_EMBED_BACKOFF_FACTOR", "1.5"))
        max_delay = float(os.getenv("RAG_EMBED_MAX_DELAY", "60"))  # Never wait more than quota window
        batch_size = int(os.getenv("RAG_EMBED_BATCH_SIZE", "16"))
        rpm_env = os.getenv("RAG_EMBED_REQUESTS_PER_MIN")
        rpm = int(rpm_env) if rpm_env else None
        quota_reset_window = float(os.getenv("RAG_EMBED_QUOTA_RESET_WINDOW", "60"))  # RPM = 60 seconds

        logger.info(
            "EmbeddingProvider initialized with model=%s, max_retries=%d, base_delay=%.1fs, "
            "max_delay=%.1fs, batch_size=%d, quota_reset_window=%.1fs",
            model, max_retries, base_delay, max_delay, batch_size, quota_reset_window
        )

        self._emb = _RetryingEmbeddings(
            raw,
            max_retries=max_retries,
            base_delay=base_delay,
            backoff_factor=backoff_factor,
            max_delay=max_delay,
            jitter=True,
            batch_size=batch_size,
            requests_per_min=rpm,
            quota_reset_window=quota_reset_window,
        )

    @property
    def embeddings(self):
        return self._emb
