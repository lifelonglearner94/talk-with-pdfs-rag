"""Simple in-memory token-bucket rate limiter used for Phase 3.

This implementation is intentionally small and test-friendly. It supports a
per-key token bucket with a fixed refill rate and capacity.
"""
import time
import threading


class TooManyRequests(Exception):
    pass


class RateLimiter:
    def __init__(self, rate: float = 1.0, capacity: int = 5):
        """Create a limiter that refills `rate` tokens per second up to `capacity`.

        rate: tokens per second
        capacity: maximum tokens bucket can hold
        """
        self.rate = float(rate)
        self.capacity = int(capacity)
        self._buckets = {}  # key -> (tokens, last_ts)
        self._lock = threading.Lock()

    def _now(self):
        return time.time()

    def consume(self, key: str, tokens: int = 1) -> bool:
        with self._lock:
            tokens = int(tokens)
            now = self._now()
            tb = self._buckets.get(key)
            if tb is None:
                self._buckets[key] = [self.capacity - tokens, now]
                if self._buckets[key][0] < 0:
                    # start with negative if initial tokens exceed capacity
                    self._buckets[key][0] = 0
                    raise TooManyRequests()
                return True
            cur_tokens, last = tb
            # refill
            delta = now - last
            refill = delta * self.rate
            cur_tokens = min(self.capacity, cur_tokens + refill)
            if cur_tokens < tokens:
                # not enough tokens
                self._buckets[key] = [cur_tokens, now]
                raise TooManyRequests()
            cur_tokens -= tokens
            self._buckets[key] = [cur_tokens, now]
            return True

    def get_tokens(self, key: str) -> float:
        with self._lock:
            tb = self._buckets.get(key)
            if tb is None:
                return float(self.capacity)
            cur_tokens, last = tb
            delta = self._now() - last
            refill = delta * self.rate
            return min(self.capacity, cur_tokens + refill)
