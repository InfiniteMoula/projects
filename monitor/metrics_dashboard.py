"""Generate a static HTML dashboard from scraper metrics."""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from utils import io as io_utils


def _load_metrics(outdir: Path) -> pd.DataFrame:
    path = outdir / "metrics" / "scraper_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found at {path}")
    return pd.read_csv(path)


def _latest_metrics(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    return {
        "enriched_pct": float(last.get("enriched_pct", 0.0)),
        "avg_time_per_site": float(last.get("avg_time_per_site", 0.0)),
        "headless_share_pct": float(last.get("headless_share_pct", 0.0)),
        "rate_4xx_pct": float(last.get("rate_4xx_pct", 0.0)),
        "rate_5xx_pct": float(last.get("rate_5xx_pct", 0.0)),
    }


def _plot_history(x: pd.Series, y: pd.Series, title: str, ylabel: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y, marker="o", color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Run")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _build_dashboard(outdir: Path, metrics_df: pd.DataFrame) -> Path:
    metrics_dir = io_utils.ensure_dir(outdir / "metrics")
    html_path = metrics_dir / "dashboard.html"
    latest = _latest_metrics(metrics_df)

    run_ids = metrics_df.get("run_id", pd.Series(range(1, len(metrics_df) + 1)))
    chart1 = _plot_history(run_ids, metrics_df.get("enriched_pct", pd.Series(dtype=float)), "Enrichment Rate (%)", "% Enriched")
    chart2 = _plot_history(run_ids, metrics_df.get("avg_time_per_site", pd.Series(dtype=float)), "Average Time per Site (s)", "Seconds")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Scraper Metrics Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f6fa; color: #2f3640; }}
        h1 {{ text-align: center; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 24px; }}
        .kpi-card {{ background: #ffffff; border-radius: 8px; padding: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }}
        .kpi-title {{ font-size: 0.85rem; text-transform: uppercase; color: #718093; }}
        .kpi-value {{ font-size: 2rem; margin-top: 8px; }}
        .charts {{ display: grid; grid-template-columns: 1fr; gap: 24px; }}
        img {{ width: 100%; border-radius: 8px; background: #fff; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>Scraper Metrics Dashboard</h1>
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">% entreprises enrichies</div>
            <div class="kpi-value">{latest['enriched_pct']:.2f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title>Temps moyen par site</div>
            <div class="kpi-value">{latest['avg_time_per_site']:.2f}s</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">% fallback headless</div>
            <div class="kpi-value">{latest['headless_share_pct']:.2f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title>Erreurs 4xx / 5xx</div>
            <div class="kpi-value">{latest['rate_4xx_pct']:.2f}% / {latest['rate_5xx_pct']:.2f}%</div>
        </div>
    </div>
    <div class="charts">
        <img src="data:image/png;base64,{chart1}" alt="Enrichment History">
        <img src="data:image/png;base64,{chart2}" alt="Average Time per Site">
    </div>
</body>
</html>
"""
    io_utils.write_text(html_path, html_content)
    return html_path


def run(outdir: Path | str) -> Path:
    outdir = Path(outdir)
    df = _load_metrics(outdir)
    return _build_dashboard(outdir, df)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate scraper metrics dashboard.")
    parser.add_argument("--outdir", required=True, help="Base output directory containing metrics.")
    args = parser.parse_args(argv)
    path = run(args.outdir)
    print(f"Dashboard generated at {path}")


if __name__ == "__main__":
    main()

