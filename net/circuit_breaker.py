from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Tuple


class CircuitBreaker:
    """Sliding window circuit breaker for tracking host failures."""

    def __init__(self, host: str, failure_threshold: float, window: int, cool_down: int) -> None:
        self.host = host
        self._threshold = max(0.0, float(failure_threshold))
        self._window = max(1, int(window))
        self._cool_down = max(0, int(cool_down))
        self._events: Deque[Tuple[float, bool]] = deque()
        self._failure_count = 0
        self._success_count = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def record_success(self) -> None:
        """Record a successful call within the window."""
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            self._events.append((now, False))
            self._success_count += 1
            if self._opened_at is not None and now - self._opened_at >= self._cool_down:
                self._opened_at = None

    def record_failure(self, reason: object | None = None) -> None:  # pragma: no cover - reason kept for debugging
        """Record a failure and potentially open the breaker."""
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            self._events.append((now, True))
            self._failure_count += 1
            if self._should_open(now):
                self._opened_at = now

    def is_open(self) -> bool:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            if self._opened_at is not None:
                if now - self._opened_at < self._cool_down:
                    return True
                # Cool down expired, allow reevaluation based on current window
                self._opened_at = None
            if self._should_open(now):
                self._opened_at = now
                return True
            return False

    def _should_open(self, now: float) -> bool:
        total = self._failure_count + self._success_count
        if total == 0:
            return False
        if self._threshold <= 0.0:
            return self._failure_count > 0
        ratio = self._failure_count / total
        return self._failure_count > 0 and ratio >= self._threshold

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._events and self._events[0][0] <= cutoff:
            _, failed = self._events.popleft()
            if failed:
                self._failure_count -= 1
            else:
                self._success_count -= 1
