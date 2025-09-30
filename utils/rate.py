"""Rate limiting helpers with jitter support."""
from __future__ import annotations

import random
import threading
import time
from typing import Dict, Tuple


def _normalize_jitter(jitter_range: Tuple[float, float]) -> Tuple[float, float]:
    start, end = jitter_range
    start = max(0.0, float(start))
    end = max(start, float(end))
    return start, end


class PerHostRateLimiter:
    """Simple per-host rate limiter with optional jitter."""

    def __init__(
        self,
        per_host_rps: float = 1.0,
        jitter_range: Tuple[float, float] = (0.2, 0.8),
    ) -> None:
        self._interval = 1.0 / per_host_rps if per_host_rps and per_host_rps > 0 else 0.0
        self._jitter = _normalize_jitter(jitter_range)
        self._lock = threading.Lock()
        self._last_request: Dict[str, float] = {}

    def wait(self, host: str) -> float:
        """Block until the caller can issue the next request for *host*."""

        now = time.monotonic()
        jitter = random.uniform(*self._jitter) if self._jitter[1] > 0 else 0.0
        with self._lock:
            last = self._last_request.get(host)
            if last is None:
                ready_time = now
            else:
                base = last + self._interval if self._interval > 0 else last
                ready_time = max(now, base)
            sleep_until = ready_time + jitter
            if self._interval > 0 and last is None:
                sleep_until = max(sleep_until, now + self._interval)
            self._last_request[host] = sleep_until
        sleep_for = max(0.0, sleep_until - now)
        if sleep_for > 0:
            time.sleep(sleep_for)
        return sleep_for


class TimeBudget:
    """Helper to track elapsed time against a budget."""

    def __init__(self, minutes: float | int | None) -> None:
        self._seconds = float(minutes) * 60.0 if minutes and minutes > 0 else 0.0
        self._start = time.monotonic()

    @property
    def exhausted(self) -> bool:
        if self._seconds <= 0:
            return False
        return (time.monotonic() - self._start) >= self._seconds

    @property
    def remaining(self) -> float:
        if self._seconds <= 0:
            return float("inf")
        elapsed = time.monotonic() - self._start
        return max(0.0, self._seconds - elapsed)


def sleep_with_jitter(jitter_range: Tuple[float, float]) -> float:
    """Sleep for a random duration within *jitter_range* seconds."""

    start, end = _normalize_jitter(jitter_range)
    if end <= 0:
        return 0.0
    duration = random.uniform(start, end)
    time.sleep(duration)
    return duration
