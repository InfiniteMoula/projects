from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from utils.metrics import record_metrics


def _resolve_output_dir(ctx: Dict[str, Any]) -> Path:
    candidate = ctx.get("outdir_path") or ctx.get("outdir") or "."
    return Path(candidate)


def run(step_cfg: Dict[str, Any] | None, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Persist aggregated metrics when the feature flag is enabled."""

    logger = ctx.get("logger")
    features = ctx.get("features", {})
    if not features.get("metrics_export", False):
        if logger:
            logger.info("Metrics export disabled; skipping step")
        return {
            "step": "monitor.metrics_export",
            "status": "SKIPPED",
            "reason": "disabled",
            "duration_s": 0,
        }

    metrics_payload = ctx.get("metrics") or {}
    base_dir = _resolve_output_dir(ctx)
    run_id = ctx.get("run_id") or "unknown"
    path = record_metrics(run_id, metrics_payload, base_dir=base_dir)

    if logger:
        logger.info("Metrics exported to %s", path)

    return {
        "step": "monitor.metrics_export",
        "status": "OK",
        "outputs": [str(path)],
    }
