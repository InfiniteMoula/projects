"""Unit tests for the adaptive controller heuristics."""

from config.enrichment_config import AdaptiveConfig
from utils.adaptive_controller import AdaptiveController


def _make_controller(initial_concurrency: int = 30, initial_chunk: int = 900) -> AdaptiveController:
    cfg = AdaptiveConfig(
        enabled=True,
        max_ram_gb=6,
        min_concurrency=10,
        max_concurrency=60,
        min_chunk=300,
        max_chunk=1200,
    )
    return AdaptiveController(
        cfg,
        initial_concurrency=initial_concurrency,
        initial_chunk_size=initial_chunk,
    )


def test_reduce_concurrency_when_errors_spike() -> None:
    controller = _make_controller()

    state = controller.observe(error_rate=0.12, req_per_min=80, ram_used=2.5)

    assert state.concurrency < 30
    assert state.concurrency_changed is True
    assert state.chunk_size == 900


def test_increase_concurrency_when_throughput_stable() -> None:
    controller = _make_controller()

    # First observation establishes the baseline throughput
    controller.observe(error_rate=0.02, req_per_min=130, ram_used=3.0)

    # Stable throughput with low error rate should trigger a 10% increase
    state = controller.observe(error_rate=0.01, req_per_min=132, ram_used=3.2)

    assert state.concurrency > 30
    assert state.concurrency_changed is True
    # Ensure we are trending towards the requested 120+ req/min target
    assert state.concurrency >= 33


def test_reduce_chunk_size_on_high_ram_usage() -> None:
    controller = _make_controller()

    state = controller.observe(error_rate=0.0, req_per_min=150, ram_used=7.5)

    assert state.chunk_size < 900
    assert state.chunk_size_changed is True
    assert state.chunk_size >= 300
