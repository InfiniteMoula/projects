"""Pipeline step exporting aggregated metrics to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from metrics.collector import get_metrics


def run(cfg: Mapping[str, object] | None, ctx: Mapping[str, object]) -> dict:
    """Export current metrics snapshot to the requested file."""

    metrics = get_metrics()
    cfg = cfg or {}
    target_path = Path(ctx.get("metrics_file") or cfg.get("target", "reports/report_metrics.json"))
    target_path = target_path.expanduser()
    try:
        export_path = metrics.export_json(target_path)
    except OSError as exc:
        return {
            "status": "WARN",
            "error": str(exc),
            "output": str(target_path),
        }

    return {
        "status": "OK",
        "output": str(export_path),
    }


__all__ = ["run"]
