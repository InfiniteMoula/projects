"""Simple metrics recording helpers."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

from utils import io


def record_metrics(run_id: str, kv: Dict[str, object], *, base_dir: Path | None = None) -> Path:
    """Append metrics for *run_id* into `metrics/scraper_stats.csv`."""

    if not run_id:
        raise ValueError("run_id must be provided")

    base = Path(base_dir) if base_dir else Path(".")
    metrics_dir = io.ensure_dir(base / "metrics")
    metrics_path = metrics_dir / "scraper_stats.csv"

    fieldnames = ["run_id"]
    fieldnames.extend(sorted(kv.keys()))

    row = {"run_id": run_id}
    row.update(kv)

    file_exists = metrics_path.exists()

    with metrics_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return metrics_path

