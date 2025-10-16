from __future__ import annotations

import pytest

from net.circuit_breaker import CircuitBreaker


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch):
    class Clock:
        def __init__(self) -> None:
            self.value = 0.0

        def now(self) -> float:
            return self.value

        def advance(self, delta: float) -> None:
            self.value += delta

        def set(self, value: float) -> None:
            self.value = value

    clock = Clock()
    monkeypatch.setattr("net.circuit_breaker.time.monotonic", clock.now)
    return clock


def test_circuit_breaker_opens_and_closes(fake_clock) -> None:
    breaker = CircuitBreaker("example.com", failure_threshold=0.5, window=10, cool_down=5)
    assert not breaker.is_open()

    breaker.record_failure()
    assert breaker.is_open()

    fake_clock.advance(4)
    assert breaker.is_open()

    fake_clock.advance(7)  # Beyond window, events are pruned
    assert not breaker.is_open()


def test_circuit_breaker_respects_cool_down(fake_clock) -> None:
    breaker = CircuitBreaker("example.com", failure_threshold=0.6, window=30, cool_down=10)

    breaker.record_failure()
    assert breaker.is_open()

    fake_clock.advance(3)
    breaker.record_success()
    assert breaker.is_open()  # Cool down still active

    fake_clock.advance(8)
    assert not breaker.is_open()


def test_circuit_breaker_sliding_window(fake_clock) -> None:
    breaker = CircuitBreaker("example.com", failure_threshold=0.5, window=5, cool_down=0)

    breaker.record_failure()
    assert breaker.is_open()

    fake_clock.advance(6)
    assert not breaker.is_open()

    breaker.record_success()
    fake_clock.advance(1)
    breaker.record_failure()
    assert breaker.is_open()

    fake_clock.advance(6)
    assert not breaker.is_open()
