from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from utils import io


def _resolve_output_dir(ctx: Dict[str, Any]) -> Path:
    candidate = ctx.get("outdir_path") or ctx.get("outdir") or "."
    return Path(candidate)


def run(step_cfg: Dict[str, Any] | None, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a lightweight correlation report when the feature is enabled."""

    logger = ctx.get("logger")
    features = ctx.get("features", {})
    if not features.get("correlation", False):
        if logger:
            logger.info("Correlation analysis disabled; skipping step")
        return {
            "step": "quality.correlation",
            "status": "SKIPPED",
            "reason": "disabled",
            "duration_s": 0,
        }

    outdir = _resolve_output_dir(ctx)
    metrics_dir = io.ensure_dir(outdir / "metrics")
    report_path = metrics_dir / "correlation.json"

    job_cfg = ctx.get("job")
    profile = job_cfg.get("profile") if isinstance(job_cfg, dict) else None

    payload = {
        "run_id": ctx.get("run_id"),
        "profile": profile,
        "inputs": step_cfg or {},
        "features": {
            key: value
            for key, value in features.items()
            if key in {"correlation", "metrics_export", "cache", "circuit_breaker"}
        },
    }
    io.write_json(report_path, payload)

    if logger:
        logger.info("Correlation report written to %s", report_path)

    return {
        "step": "quality.correlation",
        "status": "OK",
        "outputs": [str(report_path)],
    }
