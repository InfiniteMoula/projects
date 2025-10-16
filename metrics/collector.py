from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

try:  # pragma: no cover - psutil is optional at runtime
    import psutil
except Exception:  # pragma: no cover - defensive import
    psutil = None  # type: ignore[assignment]

LabelsKey = Tuple[Tuple[str, str], ...]


def _normalise_labels(labels: Optional[Mapping[str, object]]) -> LabelsKey:
    if not labels:
        return ()
    items = tuple(sorted((str(key), str(value)) for key, value in labels.items()))
    return items


def _labels_to_dict(labels: LabelsKey) -> Dict[str, str]:
    return {key: value for key, value in labels}


class Metrics:
    """In-memory metrics aggregator with JSON export capabilities."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counters: MutableMapping[str, MutableMapping[LabelsKey, float]] = defaultdict(dict)
        self._latencies: MutableMapping[str, list[float]] = defaultdict(list)
        self._created_at = time.time()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset counters and histograms to an empty state."""

        with self._lock:
            self._counters.clear()
            self._latencies.clear()
            self._created_at = time.time()

    def increment_counter(
        self,
        name: str,
        *,
        amount: float = 1.0,
        labels: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Increment a named counter by ``amount`` with optional labels."""

        key = _normalise_labels(labels)
        with self._lock:
            counter = self._counters.setdefault(name, defaultdict(float))
            counter[key] = counter.get(key, 0.0) + float(amount)

    def record_latency(
        self,
        group: str,
        duration_seconds: float,
    ) -> None:
        """Record a duration (in seconds) for the given latency group."""

        if duration_seconds < 0:
            return
        with self._lock:
            self._latencies[group].append(float(duration_seconds))

    def record_cache_hit(self, endpoint: str, *, labels: Optional[Mapping[str, object]] = None) -> None:
        base_labels = {"endpoint": endpoint}
        if labels:
            base_labels.update({str(k): str(v) for k, v in labels.items()})
        self.increment_counter("cache_hit", labels=base_labels)

    def record_cache_miss(self, endpoint: str, *, labels: Optional[Mapping[str, object]] = None) -> None:
        base_labels = {"endpoint": endpoint}
        if labels:
            base_labels.update({str(k): str(v) for k, v in labels.items()})
        self.increment_counter("cache_miss", labels=base_labels)

    def record_http_call(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_seconds: float,
        *,
        retries: int = 0,
        labels: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Record a completed HTTP call."""

        endpoint_key = endpoint or "unknown"
        base_labels: Dict[str, object] = {"endpoint": endpoint_key, "method": method.upper()}
        if labels:
            base_labels.update({str(k): str(v) for k, v in labels.items()})

        status_str = "0" if status_code <= 0 else str(int(status_code))
        base_labels.setdefault("status", status_str)

        self.increment_counter("requests_total", labels=base_labels)
        if status_code >= 400 or status_code <= 0:
            self.increment_counter("errors_total", labels=base_labels)
        if retries > 0:
            self.increment_counter("retries_total", amount=retries, labels=base_labels)

        group = str(base_labels.get("group") or endpoint_key)
        self.record_latency(group, duration_seconds)

    def snapshot(self) -> Dict[str, object]:
        """Return a snapshot of counters, latency distributions and resources."""

        with self._lock:
            counters_payload = {
                name: [
                    {"labels": _labels_to_dict(labels), "value": value}
                    for labels, value in sorted(values.items())
                ]
                for name, values in self._counters.items()
            }

            latency_payload = {
                group: self._describe_latencies(samples)
                for group, samples in self._latencies.items()
                if samples
            }

        resources: Dict[str, float] = {}
        if psutil is not None:  # pragma: no branch - guarded import
            try:
                process = psutil.Process()
                with process.oneshot():
                    try:
                        rss = float(process.memory_info().rss)
                        resources["ram_mb"] = round(rss / (1024 * 1024), 3)
                    except Exception:  # pragma: no cover - defensive
                        pass
                    try:
                        cpu_percent = float(psutil.cpu_percent(interval=None))
                        resources["cpu_percent"] = cpu_percent
                    except Exception:  # pragma: no cover - defensive
                        pass
            except Exception:  # pragma: no cover - defensive
                pass

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": round(time.time() - self._created_at, 3),
            "counters": counters_payload,
            "latency": latency_payload,
            "resources": resources,
        }

    def export_json(self, path: str | Path = "reports/report_metrics.json") -> Path:
        """Export the current snapshot to a JSON file."""

        snapshot = self.snapshot()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    def summary(self) -> Dict[str, object]:
        """Return aggregated KPI figures useful for CLI reporting."""

        with self._lock:
            total_requests = sum(self._counters.get("requests_total", {}).values())
            total_errors = sum(self._counters.get("errors_total", {}).values())
            total_retries = sum(self._counters.get("retries_total", {}).values())
            cache_hits = sum(self._counters.get("cache_hit", {}).values())
            cache_misses = sum(self._counters.get("cache_miss", {}).values())

            latency_counts = {group: len(samples) for group, samples in self._latencies.items() if samples}
            latency_totals = {group: sum(samples) for group, samples in self._latencies.items() if samples}

        total_latency_count = sum(latency_counts.values())
        total_latency_sum = sum(latency_totals.values())
        avg_latency_ms = (total_latency_sum / total_latency_count * 1000) if total_latency_count else 0.0

        top_latency_groups = []
        with self._lock:
            for group, samples in self._latencies.items():
                if not samples:
                    continue
                stats = self._describe_latencies(samples)
                top_latency_groups.append((group, stats["p90_ms"]))
        top_latency_groups.sort(key=lambda item: item[1], reverse=True)
        top_latency_groups = top_latency_groups[:3]

        error_rate = (total_errors / total_requests) if total_requests else 0.0

        return {
            "total_requests": int(total_requests),
            "total_errors": int(total_errors),
            "error_rate": round(error_rate, 4),
            "total_retries": int(total_retries),
            "cache_hit": int(cache_hits),
            "cache_miss": int(cache_misses),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "top_latency_groups": [
                {"group": group, "p90_ms": round(p90, 2)} for group, p90 in top_latency_groups
            ],
        }

    def format_summary(self) -> str:
        """Return a compact, human readable KPI summary line."""

        data = self.summary()
        parts = [
            f"requests={data['total_requests']}",
            f"errors={data['total_errors']} (rate={data['error_rate']:.2%})",
            f"retries={data['total_retries']}",
            f"cache={data['cache_hit']}/{data['cache_miss']} (hit/miss)",
            f"avg_latency_ms={data['avg_latency_ms']}",
        ]
        if data["top_latency_groups"]:
            top = ", ".join(
                f"{entry['group']}@p90={entry['p90_ms']}ms" for entry in data["top_latency_groups"]
            )
            parts.append(f"hotspots=[{top}]")
        return "KPI SUMMARY | " + " | ".join(parts)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _describe_latencies(self, samples: Iterable[float]) -> Dict[str, float]:
        values = sorted(float(sample) for sample in samples if sample >= 0)
        if not values:
            return {"count": 0, "min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0}

        count = len(values)
        total = sum(values)
        return {
            "count": count,
            "min_ms": round(values[0] * 1000, 3),
            "max_ms": round(values[-1] * 1000, 3),
            "avg_ms": round((total / count) * 1000, 3),
            "p50_ms": round(self._percentile(values, 0.50) * 1000, 3),
            "p90_ms": round(self._percentile(values, 0.90) * 1000, 3),
            "p99_ms": round(self._percentile(values, 0.99) * 1000, 3),
        }

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        k = (len(values) - 1) * percentile
        lower = math.floor(k)
        upper = math.ceil(k)
        if lower == upper:
            return values[int(k)]
        lower_value = values[lower]
        upper_value = values[upper]
        return lower_value + (upper_value - lower_value) * (k - lower)


_GLOBAL_METRICS = Metrics()


def get_metrics() -> Metrics:
    """Return the global metrics singleton used across the pipeline."""

    return _GLOBAL_METRICS
