"""Compute enrichment score and generate report.""" 
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from utils import io

CONTACTABILITY_MAX = 65.0  # 35 email + 20 phone + 10 site
CONTACTABILITY_WEIGHT = 55.0
COMPLETENESS_MAX = 30.0  # address + siret + social
COMPLETENESS_WEIGHT = 20.0
UNICITY_WEIGHT = 15.0
FRESHNESS_WEIGHT = 10.0
FRESHNESS_THRESHOLD_DAYS = 14


def _as_list(value) -> List[str]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, 'tolist'):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return converted
            if isinstance(converted, tuple):
                return list(converted)
            if converted is None:
                return []
            return [converted]
        except Exception:
            pass
    if isinstance(value, str):
        return [value]
    return []


def _as_dict(value) -> Dict[str, List[str]]:
    if isinstance(value, dict):
        normalized: Dict[str, List[str]] = {}
        for key, raw in value.items():
            if raw is None:
                normalized[key] = []
                continue
            if hasattr(raw, 'tolist'):
                try:
                    raw = raw.tolist()
                except Exception:
                    pass
            if isinstance(raw, list):
                normalized[key] = raw
            elif isinstance(raw, tuple):
                normalized[key] = list(raw)
            elif isinstance(raw, set):
                normalized[key] = list(raw)
            elif raw is None:
                normalized[key] = []
            else:
                normalized[key] = [raw]
        return normalized
    return {}


def _coverage_ratio(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean()) * 100.0


def _format_pct(value: float) -> str:
    return f"{value:.1f}%"


def _build_histogram(scores: Iterable[float]) -> str:
    data = list(scores)
    if not data:
        return "<p>No scores available.</p>"
    buckets = [0] * 10
    for score in data:
        idx = min(9, max(0, int(score // 10)))
        buckets[idx] += 1
    max_height = max(buckets) or 1
    width = 360
    height = 120
    bar_width = width // len(buckets)
    svg_parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Score distribution">'
    ]
    for idx, count in enumerate(buckets):
        bar_height = int((count / max_height) * (height - 20))
        x_pos = idx * bar_width
        y_pos = height - bar_height - 20
        svg_parts.append(
            f'<rect x="{x_pos}" y="{y_pos}" width="{bar_width - 4}" height="{bar_height}" fill="#2F80ED" />'
        )
        label = f"{idx * 10}-{idx * 10 + 9}"
        svg_parts.append(
            f'<text x="{x_pos + bar_width / 2}" y="{height - 5}" font-size="10" text-anchor="middle">{label}</text>'
        )
    svg_parts.append("</svg>")
    return "".join(svg_parts)


def run(cfg: dict, ctx: dict) -> dict:
    logger = ctx.get("logger")
    outdir = Path(ctx["outdir"])
    contacts_path = outdir / "contacts" / "contacts.parquet"
    if not contacts_path.exists():
        if logger:
            logger.warning("enrich_score: contacts.parquet not found")
        return {"status": "SKIPPED", "reason": "NO_CONTACTS"}

    df = pd.read_parquet(contacts_path)
    if df.empty:
        return {"status": "SKIPPED", "reason": "EMPTY_CONTACTS"}

    now = pd.Timestamp.utcnow()

    email_type_series = df.get("email_type")
    if email_type_series is None:
        email_type_series = pd.Series(["" for _ in range(len(df))], index=df.index)
    else:
        email_type_series = email_type_series.fillna("")

    email_any = df["emails"].apply(lambda v: bool(_as_list(v)))
    email_nominative = email_type_series.str.lower() == "nominative"
    email_generic = email_type_series.str.lower() == "generic"
    phone_any = df["phones"].apply(lambda v: bool(_as_list(v)))
    social_any = df["social_links"].apply(lambda v: any(_as_list(link) for link in _as_dict(v).values()))
    address_any = df["addresses"].apply(lambda v: bool(_as_list(v)))
    siret_any = df["sirets"].apply(lambda v: bool(_as_list(v)))

    site_ok = (
        df.get("best_status", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int).between(200, 399)
        & df.get("top_url", pd.Series(["" for _ in range(len(df))], index=df.index)).fillna("").str.startswith("https")
    )

    email_any_int = email_any.astype(int)
    email_nominative_int = email_nominative.astype(int)
    phone_int = phone_any.astype(int)
    site_int = site_ok.astype(int)
    social_int = social_any.astype(int)
    address_int = address_any.astype(int)
    siret_int = siret_any.astype(int)

    contactability_base = (
        email_nominative_int * 35
        + (email_any_int - email_nominative_int).clip(lower=0) * 20
        + phone_int * 20
        + site_int * 10
    )
    contactability = (contactability_base / CONTACTABILITY_MAX) * CONTACTABILITY_WEIGHT

    completeness_base = (
        address_int * 10
        + siret_int * 10
        + social_int * 10
    )
    completeness = (completeness_base / COMPLETENESS_MAX) * COMPLETENESS_WEIGHT

    primary_siret = df["sirets"].apply(lambda v: _as_list(v)[0] if _as_list(v) else None)
    siret_counts = primary_siret.value_counts(dropna=True)
    best_email_counts = df["best_email"].value_counts(dropna=True)
    domain_counts = df["domain"].value_counts(dropna=True)

    unicity_scores = []
    freshness_scores = []
    discovered_at_series = pd.to_datetime(df.get("discovered_at"), errors="coerce", utc=True)

    for idx, row in df.iterrows():
        domain = row.get("domain")
        best_email = row.get("best_email")
        siret = primary_siret.iloc[idx]

        unique_score = 0
        if pd.notna(siret) and siret_counts.get(siret, 0) <= 1:
            unique_score += 5
        if isinstance(best_email, str) and best_email and best_email_counts.get(best_email, 0) <= 1:
            unique_score += 5
        has_contact = bool(email_any.iloc[idx] or phone_any.iloc[idx])
        if has_contact and isinstance(domain, str) and domain and domain_counts.get(domain, 0) <= 1:
            unique_score += 5
        unicity_scores.append(unique_score)

        discovered_at = discovered_at_series.iloc[idx]
        if pd.notna(discovered_at) and (now - discovered_at).days <= FRESHNESS_THRESHOLD_DAYS:
            freshness_scores.append(FRESHNESS_WEIGHT)
        else:
            freshness_scores.append(0.0)

    df["contactability_score"] = contactability.round(2)
    df["completeness_score"] = completeness.round(2)
    df["unicity_score"] = pd.Series(unicity_scores, index=df.index, dtype=float)
    df["freshness_score"] = pd.Series(freshness_scores, index=df.index, dtype=float)
    df["enrich_score"] = (
        df["contactability_score"]
        + df["completeness_score"]
        + df["unicity_score"]
        + df["freshness_score"]
    ).round(2)

    df.to_parquet(contacts_path, index=False)

    coverage_stats = {
        "email_any": _format_pct(_coverage_ratio(email_any)),
        "email_nominative": _format_pct(_coverage_ratio(email_nominative)),
        "email_generic": _format_pct(_coverage_ratio(email_generic)),
        "phone_any": _format_pct(_coverage_ratio(phone_any)),
        "site_ok": _format_pct(_coverage_ratio(site_ok)),
        "address": _format_pct(_coverage_ratio(address_any)),
        "siret": _format_pct(_coverage_ratio(siret_any)),
        "social": _format_pct(_coverage_ratio(social_any)),
    }

    histogram_svg = _build_histogram(df["enrich_score"].fillna(0))

    top_domains = df.sort_values("enrich_score", ascending=False).head(10)
    rows_html = "".join(
        f"<tr><td>{idx + 1}</td><td>{row['domain']}</td><td>{row['enrich_score']:.1f}</td>"
        f"<td>{row.get('best_email') or ''}</td><td>{', '.join(_as_list(row.get('phones'))[:2])}</td></tr>"
        for idx, row in top_domains.iterrows()
    )

    coverage_rows = "".join(
        f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in coverage_stats.items()
    )

    report_html = f"""
    <!DOCTYPE html>
    <html lang=\"fr\">
    <head>
      <meta charset=\"utf-8\" />
      <title>Rapport Enrichissement</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; color: #1b1f23; }}
        h1, h2 {{ color: #0b5394; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
        th, td {{ border: 1px solid #d0d7de; padding: 8px; text-align: left; }}
        th {{ background-color: #f6f8fa; }}
        .metrics {{ width: 320px; }}
      </style>
    </head>
    <body>
      <h1>Rapport d'enrichissement</h1>
      <p>Genere le {io.now_iso()}</p>
      <h2>Couverture</h2>
      <table class=\"metrics\">{coverage_rows}</table>
      <h2>Distribution des scores</h2>
      {histogram_svg}
      <h2>Top domaines</h2>
      <table>
        <tr><th>#</th><th>Domaine</th><th>Score</th><th>Email</th><th>Telephone</th></tr>
        {rows_html}
      </table>
    </body>
    </html>
    """

    reports_dir = io.ensure_dir(outdir / "reports")
    report_path = reports_dir / "enrichment_report.html"
    io.write_text(report_path, report_html)

    return {
        "status": "OK",
        "records": len(df),
        "report": str(report_path),
        "average_score": float(df["enrich_score"].mean()),
    }
