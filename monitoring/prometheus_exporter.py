"""
Lightweight Prometheus exporter used by ``builder_cli``.

The exporter is intentionally dependency-free unless ``prometheus_client`` is
installed.  All functions become no-ops when the library is missing so the
pipeline keeps working without Prometheus.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _PROM_AVAILABLE = False

__all__ = [
    "PROMETHEUS_AVAILABLE",
    "start_metrics_server",
    "set_run_metadata",
    "observe_step",
]


LOG = logging.getLogger("monitoring.prometheus")
PROMETHEUS_AVAILABLE = _PROM_AVAILABLE

_SERVER_STARTED = False
_SERVER_LOCK = threading.Lock()

if _PROM_AVAILABLE:  # pragma: no cover - guarded initialisation
    _STEP_DURATION = Histogram(
        "pipeline_step_duration_seconds",
        "Observed execution time per pipeline step",
        labelnames=("step",),
    )
    _STEP_STATUS = Counter(
        "pipeline_step_status_total",
        "Step execution status count",
        labelnames=("step", "status"),
    )
    _STEP_LAST_DURATION = Gauge(
        "pipeline_step_last_duration_seconds",
        "Duration of the most recent completed execution per step",
        labelnames=("step",),
    )
    _STEP_LAST_TS = Gauge(
        "pipeline_step_completed_timestamp_seconds",
        "Unix timestamp of the last completed execution per step",
        labelnames=("step",),
    )
    _RUN_INFO = Gauge(
        "pipeline_run_info",
        "Static metadata about the current pipeline run. "
        "Value equals the number of steps scheduled for execution.",
        labelnames=("run_id", "job", "profile"),
    )
else:  # pragma: no cover - no dependency
    _STEP_DURATION = None
    _STEP_STATUS = None
    _STEP_LAST_DURATION = None
    _STEP_LAST_TS = None
    _RUN_INFO = None


def start_metrics_server(port: int, address: str = "0.0.0.0") -> bool:
    """
    Start the Prometheus HTTP server if the client library is available.
    """

    if not _PROM_AVAILABLE:
        LOG.warning(
            "prometheus_client not installed; metrics exporter remains disabled"
        )
        return False

    if port <= 0:
        return False

    global _SERVER_STARTED
    with _SERVER_LOCK:
        if _SERVER_STARTED:
            return True
        start_http_server(port, addr=address)
        _SERVER_STARTED = True
        LOG.info("Prometheus metrics server listening on %s:%s", address, port)
        return True


def set_run_metadata(run_id: str, job_name: str, profile: Optional[str], total_steps: int) -> None:
    """
    Record run-level metadata so dashboards can tie metrics to a specific run.
    """

    if not _PROM_AVAILABLE or not _SERVER_STARTED:
        return
    _RUN_INFO.labels(
        run_id=run_id,
        job=job_name,
        profile=profile or "unknown",
    ).set(float(total_steps))


def observe_step(step_name: str, status: str, duration: float) -> None:
    """
    Update Prometheus time series for a completed step.
    """

    if not _PROM_AVAILABLE or not _SERVER_STARTED:
        return

    duration_value = max(float(duration), 0.0)
    now = time.time()
    _STEP_STATUS.labels(step=step_name, status=status).inc()
    _STEP_DURATION.labels(step=step_name).observe(duration_value)
    _STEP_LAST_DURATION.labels(step=step_name).set(duration_value)
    _STEP_LAST_TS.labels(step=step_name).set(now)
