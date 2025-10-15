from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import psutil
import yaml

import builder_cli


@dataclass
class CoverageMetrics:
    coverage: float
    avg_score: float


def _load_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return None


def _compute_coverage(df: Optional[pd.DataFrame], value_col: str, score_col: Optional[str] = None) -> CoverageMetrics:
    if df is None or df.empty or value_col not in df.columns:
        return CoverageMetrics(coverage=0.0, avg_score=0.0)
    values = df[value_col].fillna("").astype("string").str.strip()
    coverage = float((values != "").sum()) / float(len(df))
    avg_score = 0.0
    if score_col and score_col in df.columns:
        avg_score = float(df[score_col].fillna(0.0).astype("float64").mean())
    return CoverageMetrics(coverage=coverage, avg_score=avg_score)


def _extract_budget_stats(log_path: Path) -> Dict[str, float]:
    stats: Dict[str, float] = {"http_requests": 0.0, "http_bytes": 0.0}
    if not log_path.exists():
        return stats
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                budget = payload.get("budget_stats") or {}
                stats["http_requests"] = max(stats["http_requests"], float(budget.get("http_requests", 0.0)))
                stats["http_bytes"] = max(stats["http_bytes"], float(budget.get("http_bytes", 0.0)))
    except OSError:
        pass
    return stats


def run_pipeline(sample_path: Path, outdir: Path, profile: str) -> Dict[str, float]:
    outdir.mkdir(parents=True, exist_ok=True)
    job_path = outdir / "job_standard_nocapital.yaml"
    job_path.write_text(yaml.safe_dump({"profile": profile}, allow_unicode=True), encoding="utf-8")

    argv = [
        "run-profile",
        "--job",
        str(job_path),
        "--out",
        str(outdir),
        "--input",
        str(sample_path),
        "--profile",
        profile,
    ]

    start = time.time()
    rc = builder_cli.main(argv)
    duration = time.time() - start
    if rc != 0:
        raise SystemExit(f"Pipeline execution failed with exit code {rc}")

    logs_dir = outdir / "logs"
    log_files = list(logs_dir.glob("*.json"))
    budget_stats = {"http_requests": 0.0, "http_bytes": 0.0}
    if log_files:
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        budget_stats = _extract_budget_stats(latest_log)
    return {"duration_s": duration, **budget_stats}


def compute_kpis(outdir: Path, duration_s: float) -> Dict[str, float]:
    domains_df = _load_table(outdir / "domains_enriched.parquet")
    contacts_df = _load_table(outdir / "contacts_enriched.parquet")
    linkedin_df = _load_table(outdir / "linkedin_enriched.parquet")

    site_metrics = _compute_coverage(domains_df, "site_web", "site_web_score")
    email_metrics = _compute_coverage(contacts_df, "email", "email_score")
    phone_metrics = _compute_coverage(contacts_df, "telephone", "telephone_score")
    linkedin_metrics = _compute_coverage(linkedin_df, "linkedin_url", "linkedin_score")

    total_rows = len(domains_df) if domains_df is not None else 0
    return {
        "rows": total_rows,
        "site_coverage": site_metrics.coverage,
        "site_score_avg": site_metrics.avg_score,
        "email_coverage": email_metrics.coverage,
        "email_score_avg": email_metrics.avg_score,
        "phone_coverage": phone_metrics.coverage,
        "phone_score_avg": phone_metrics.avg_score,
        "linkedin_coverage": linkedin_metrics.coverage,
        "linkedin_score_avg": linkedin_metrics.avg_score,
        "duration_s": duration_s,
    }


def print_summary(results: Dict[str, float]) -> None:
    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    req_per_min = 0.0
    if results.get("http_requests", 0.0) and results.get("duration_s", 0.0):
        req_per_min = (results["http_requests"] / results["duration_s"]) * 60.0

    print("\n=== Standard No Capital KPI Summary ===")
    print(f"Rows processed        : {int(results.get('rows', 0))}")
    print(f"Web coverage          : {results.get('site_coverage', 0.0):.1%} (avg score {results.get('site_score_avg', 0.0):.2f})")
    print(f"Email coverage        : {results.get('email_coverage', 0.0):.1%} (avg score {results.get('email_score_avg', 0.0):.2f})")
    print(f"Phone coverage        : {results.get('phone_coverage', 0.0):.1%} (avg score {results.get('phone_score_avg', 0.0):.2f})")
    print(f"LinkedIn coverage     : {results.get('linkedin_coverage', 0.0):.1%} (avg score {results.get('linkedin_score_avg', 0.0):.2f})")
    print(f"Total duration        : {results.get('duration_s', 0.0):.1f}s")
    if req_per_min:
        print(f"HTTP requests/min     : {req_per_min:.1f}")
    if results.get("http_bytes"):
        print(f"HTTP bytes transferred: {results['http_bytes'] / (1024 * 1024):.2f} MB")
    print(f"Resident memory usage : {rss_mb:.1f} MB\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standard_nocapital profile on a sample parquet and report KPIs.")
    parser.add_argument("--sample", type=Path, default=Path("sample_50.parquet"), help="Input parquet sample")
    parser.add_argument("--out", type=Path, default=Path("out/standard_nocapital_sample"), help="Output directory")
    parser.add_argument("--profile", default="standard_nocapital", help="Profile to execute")
    args = parser.parse_args()

    if not args.sample.exists():
        raise FileNotFoundError(f"Sample parquet not found: {args.sample}")

    run_stats = run_pipeline(args.sample.resolve(), args.out.resolve(), args.profile)
    kpis = compute_kpis(args.out.resolve(), run_stats["duration_s"])

    combined = {**run_stats, **kpis}
    print_summary(combined)


if __name__ == "__main__":
    main()
