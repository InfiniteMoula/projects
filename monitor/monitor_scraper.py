"""Monitor scraper outputs and record aggregate metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from utils import io
from utils.metrics import record_metrics


def _load_contacts(outdir: Path) -> pd.DataFrame:
    contacts_clean = outdir / "contacts" / "contacts_clean.parquet"
    if contacts_clean.exists():
        return pd.read_parquet(contacts_clean)
    contacts_raw = outdir / "contacts" / "contacts.parquet"
    if contacts_raw.exists():
        return pd.read_parquet(contacts_raw)
    return pd.DataFrame()


def _load_no_contact(outdir: Path) -> pd.DataFrame:
    path = outdir / "contacts" / "no_contact.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _load_scraper_log(outdir: Path) -> List[Dict[str, object]]:
    log_path = outdir / "logs" / "scraper.jsonl"
    entries: List[Dict[str, object]] = []
    if not log_path.exists():
        return entries
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return entries
    return entries


def _aggregate_contacts(df: pd.DataFrame) -> Tuple[int, int, float]:
    if df.empty:
        return 0, 0, 0.0
    total = len(df)
    enriched = int((df["emails"].apply(bool) | df["phones"].apply(bool)).sum())
    percentage = (enriched / total) * 100 if total else 0.0
    return total, enriched, percentage


def _status_rates(df: pd.DataFrame) -> Tuple[float, float]:
    if df.empty or "best_status" not in df.columns:
        return 0.0, 0.0
    status_counts = df["best_status"].dropna().astype(int)
    total = len(status_counts)
    if total == 0:
        return 0.0, 0.0
    rate_4xx = (status_counts.between(400, 499).sum() / total) * 100
    rate_5xx = (status_counts.between(500, 599).sum() / total) * 100
    return rate_4xx, rate_5xx


def _headless_share(df: pd.DataFrame) -> float:
    if df.empty or "best_page" not in df.columns:
        return 0.0
    headless_pages = df["best_page"].astype(str).str.contains("headless", case=False, na=False)
    total = len(df)
    if total == 0:
        return 0.0
    return (headless_pages.sum() / total) * 100


def _time_per_site(logs: List[Dict[str, object]]) -> float:
    durations = [entry.get("duration") for entry in logs if isinstance(entry.get("duration"), (int, float))]
    if not durations:
        return 0.0
    return sum(durations) / len(durations)


def run(cfg: dict, ctx: dict) -> dict:
    logger = ctx.get("logger")
    outdir = Path(ctx["outdir"])
    contacts_df = _load_contacts(outdir)
    no_contact_df = _load_no_contact(outdir)
    scraper_logs = _load_scraper_log(outdir)

    total_companies, enriched_companies, enriched_pct = _aggregate_contacts(contacts_df)
    rate_4xx, rate_5xx = _status_rates(contacts_df)
    headless_pct = _headless_share(contacts_df)
    average_time = _time_per_site(scraper_logs)

    fallback_total = len(no_contact_df) if not no_contact_df.empty else 0

    summary = {
        "total_companies": total_companies,
        "enriched_companies": enriched_companies,
        "enriched_pct": round(enriched_pct, 2),
        "avg_time_per_site": round(average_time, 3),
        "rate_4xx_pct": round(rate_4xx, 2),
        "rate_5xx_pct": round(rate_5xx, 2),
        "headless_share_pct": round(headless_pct, 2),
        "fallback_domains": fallback_total,
    }

    run_id = str(ctx.get("run_id") or cfg.get("run_id") or io.now_iso())
    record_metrics(run_id, summary, base_dir=outdir)

    metrics_dir = io.ensure_dir(outdir / "metrics")
    summary_path = metrics_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    message = (
        f"monitor_scraper: {enriched_companies}/{total_companies} enriched "
        f"({summary['enriched_pct']}%), headless share {summary['headless_share_pct']}%, "
        f"4xx {summary['rate_4xx_pct']}%, 5xx {summary['rate_5xx_pct']}%"
    )
    if logger:
        logger.info(message)
    else:
        print(message)

    return {
        "status": "OK",
        "metrics_csv": str(outdir / "metrics" / "scraper_stats.csv"),
        "summary": summary,
        "summary_path": str(summary_path),
    }

