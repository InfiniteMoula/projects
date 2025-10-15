import json

import pytest

from metrics.collector import Metrics


def test_counters_and_labels_accumulate() -> None:
    metrics = Metrics()
    metrics.increment_counter("requests_total", labels={"endpoint": "api", "method": "GET"})
    metrics.increment_counter("requests_total", labels={"endpoint": "api", "method": "GET"})
    metrics.increment_counter("errors_total", labels={"endpoint": "api", "method": "GET"})
    snapshot = metrics.snapshot()

    assert "requests_total" in snapshot["counters"]
    entry = snapshot["counters"]["requests_total"][0]
    assert entry["labels"] == {"endpoint": "api", "method": "GET"}
    assert entry["value"] == pytest.approx(2)

    error_entry = snapshot["counters"]["errors_total"][0]
    assert error_entry["value"] == pytest.approx(1)


def test_latency_percentiles_and_summary() -> None:
    metrics = Metrics()
    samples = [0.10, 0.20, 0.30, 0.40, 0.50]
    for sample in samples:
        metrics.record_latency("service", sample)
    metrics.record_cache_hit("service")
    metrics.record_cache_miss("service")
    summary = metrics.summary()

    assert summary["cache_hit"] == 1
    assert summary["cache_miss"] == 1

    snapshot = metrics.snapshot()
    stats = snapshot["latency"]["service"]
    assert stats["count"] == len(samples)
    assert stats["p50_ms"] == pytest.approx(300, abs=1)
    assert stats["p90_ms"] == pytest.approx(460, abs=1)
    assert stats["p99_ms"] == pytest.approx(496, abs=1)


def test_export_json_creates_report(tmp_path) -> None:
    metrics = Metrics()
    metrics.increment_counter("requests_total", labels={"endpoint": "api", "method": "GET"})
    metrics.record_latency("api", 0.123)

    path = tmp_path / "report.json"
    metrics.export_json(path)

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "counters" in data and "requests_total" in data["counters"]
    assert data["counters"]["requests_total"][0]["value"] == pytest.approx(1)
    assert data["latency"]["api"]["p50_ms"] == pytest.approx(123, abs=1)
