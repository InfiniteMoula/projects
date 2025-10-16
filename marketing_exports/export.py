"""Utilities to generate marketing-ready exports from enriched datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
import html

import pandas as pd

from ml.lead_score import add_business_score
from quality.validation import validate_email, validate_telephone
from utils import io


@dataclass(frozen=True)
class MarketingKpis:
    total_leads: int
    contactable_leads: int
    valid_email_rate: float
    duplicate_rate: float
    duplicate_key: Optional[str]
    top_naf: Sequence[tuple[str, int]]


def _load_dataset(outdir: Path, dataset_path: Optional[Path]) -> pd.DataFrame:
    """Load the enriched dataset from *outdir* or an explicit *dataset_path*."""

    if dataset_path is not None:
        path = dataset_path
    else:
        parquet = outdir / "dataset_enriched.parquet"
        csv = outdir / "dataset_enriched.csv"
        if parquet.exists():
            path = parquet
        elif csv.exists():
            path = csv
        else:
            raise FileNotFoundError(
                "Unable to locate dataset_enriched.[parquet|csv]; provide --dataset"
            )

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _ensure_score(df: pd.DataFrame) -> pd.DataFrame:
    if "score_business" in df.columns and not df["score_business"].isna().all():
        df["score_business"] = pd.to_numeric(df["score_business"], errors="coerce").fillna(0.0)
        return df
    scored = add_business_score(df, inplace=False)
    scored["score_business"] = pd.to_numeric(
        scored["score_business"], errors="coerce"
    ).fillna(0.0)
    return scored


def _boolean_series(series: pd.Series) -> pd.Series:
    if series.dtype == "boolean":
        return series.fillna(False)
    return series.astype(bool)


def _compute_email_valid(df: pd.DataFrame) -> pd.Series:
    if "email_valid" in df.columns:
        return _boolean_series(df["email_valid"])
    if "email" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df["email"].fillna("").astype(str).map(
        lambda value: bool(validate_email(value, check_mx=False).is_valid)
    )


def _compute_phone_valid(df: pd.DataFrame) -> pd.Series:
    if "telephone_valid" in df.columns:
        return _boolean_series(df["telephone_valid"])
    candidates = [
        col
        for col in ("telephone", "telephone_norm", "phone", "telephone_e164")
        if col in df.columns
    ]
    if not candidates:
        return pd.Series([False] * len(df), index=df.index)
    series = df[candidates[0]].fillna("").astype(str)
    return series.map(lambda value: bool(validate_telephone(value).is_valid))


def _choose_identifier_column(df: pd.DataFrame) -> Optional[str]:
    for column in (
        "siren",
        "siret",
        "siren_key",
        "siret_key",
        "siren_norm",
        "siret_norm",
        "domain",
        "company_id",
    ):
        if column in df.columns:
            series = df[column].fillna("").astype(str).str.strip()
            if series.ne("").any():
                return column
    return None


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    identifier = _choose_identifier_column(df)
    if not identifier:
        return df
    normalized = df[identifier].fillna("").astype(str).str.strip().str.lower()
    unique_mask = ~normalized.duplicated(keep="first")
    return df.loc[unique_mask].copy()


def _coalesce(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return df[column]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _prepare_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    export_df = pd.DataFrame(
        {
            "company_name": _coalesce(
                df,
                (
                    "company_name",
                    "name",
                    "denomination_usuelle",
                    "Nom entreprise",
                ),
            ),
            "siren": _coalesce(df, ("siren",)),
            "siret": _coalesce(df, ("siret",)),
            "naf": _coalesce(df, ("naf", "naf_code", "ape")),
            "region": _coalesce(df, ("region", "region_label", "region_name")),
            "email": _coalesce(df, ("email", "best_email")),
            "telephone": _coalesce(
                df, ("telephone", "telephone_norm", "phone", "telephone_e164")
            ),
            "site_web": _coalesce(df, ("site_web", "website", "domain")),
            "score_business": pd.to_numeric(
                _coalesce(df, ("score_business",)), errors="coerce"
            ).fillna(0.0),
            "email_valid": _coalesce(df, ("email_valid",)),
            "telephone_valid": _coalesce(df, ("telephone_valid",)),
        }
    )
    return export_df


def _compute_kpis(
    df: pd.DataFrame,
    email_valid: pd.Series,
    phone_valid: pd.Series,
) -> MarketingKpis:
    total_leads = int(len(df))
    contactable = int((email_valid | phone_valid).sum())
    valid_email_rate = round(
        float(email_valid.astype(int).sum() * 100.0 / total_leads) if total_leads else 0.0,
        2,
    )

    identifier = _choose_identifier_column(df)
    duplicate_rate = 0.0
    duplicate_key: Optional[str] = None
    if identifier:
        series = df[identifier].fillna("").astype(str).str.strip().str.lower()
        non_empty = series.ne("")
        duplicates = series.duplicated(keep=False) & non_empty
        duplicate_rate = round(
            float(duplicates.sum() * 100.0 / total_leads) if total_leads else 0.0,
            2,
        )
        if duplicates.any():
            duplicate_key = identifier

    naf_column = None
    for candidate in ("naf", "naf_code", "ape"):
        if candidate in df.columns:
            naf_column = candidate
            break

    top_naf: list[tuple[str, int]] = []
    if naf_column is not None:
        naf_series = (
            df[naf_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"": pd.NA})
            .dropna()
        )
        counts = naf_series.value_counts().head(5)
        top_naf = [(code, int(count)) for code, count in counts.items()]

    return MarketingKpis(
        total_leads=total_leads,
        contactable_leads=contactable,
        valid_email_rate=valid_email_rate,
        duplicate_rate=duplicate_rate,
        duplicate_key=duplicate_key,
        top_naf=tuple(top_naf),
    )


def _render_dashboard_html(
    export_dir: Path,
    kpis: MarketingKpis,
    sample: pd.DataFrame,
) -> Path:
    rows = []
    for _, row in sample.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('company_name', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('siren', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('naf', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('region', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('email', '') or ''))}</td>"
            f"<td>{html.escape(str(row.get('telephone', '') or ''))}</td>"
            f"<td>{row.get('score_business', 0):.2f}</td>"
            "</tr>"
        )

    top_naf_html = "".join(
        f"<li><strong>{html.escape(code)}</strong> — {count} entreprises</li>"
        for code, count in kpis.top_naf
    ) or "<li>Aucun code disponible</li>"

    duplicate_label = (
        f"{kpis.duplicate_rate:.2f}% (clé: {kpis.duplicate_key})"
        if kpis.duplicate_key
        else f"{kpis.duplicate_rate:.2f}%"
    )

    html_content = f"""<!DOCTYPE html>
<html lang='fr'>
<head>
  <meta charset='utf-8'>
  <title>Dashboard marketing B2B</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 24px; color: #1b1f23; }}
    h1 {{ font-size: 24px; margin-bottom: 16px; }}
    .kpis {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; }}
    .kpi {{ background: #f6f8fa; padding: 16px 20px; border-radius: 8px; min-width: 180px; box-shadow: 0 1px 2px rgba(31,35,40,0.08); }}
    .kpi .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: #57606a; }}
    .kpi .value {{ display: block; font-size: 22px; font-weight: 600; margin-top: 8px; color: #24292f; }}
    ol {{ padding-left: 20px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 24px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #f0f6ff; font-weight: 600; color: #1b1f23; }}
    tbody tr:nth-child(even) {{ background: #f6f8fa; }}
  </style>
</head>
<body>
  <h1>Orientation marketing B2B</h1>
  <div class='kpis'>
    <div class='kpi'>
      <span class='label'>Total leads</span>
      <span class='value'>{kpis.total_leads}</span>
    </div>
    <div class='kpi'>
      <span class='label'>Leads contactables</span>
      <span class='value'>{kpis.contactable_leads}</span>
    </div>
    <div class='kpi'>
      <span class='label'>Taux d'emails valides</span>
      <span class='value'>{kpis.valid_email_rate:.2f}%</span>
    </div>
    <div class='kpi'>
      <span class='label'>Taux de doublons</span>
      <span class='value'>{duplicate_label}</span>
    </div>
  </div>
  <h2>Top 5 codes NAF</h2>
  <ol>
    {top_naf_html}
  </ol>
  <h2>Top entreprises contactables</h2>
  <table>
    <thead>
      <tr>
        <th>Entreprise</th>
        <th>SIREN</th>
        <th>NAF</th>
        <th>Région</th>
        <th>Email</th>
        <th>Téléphone</th>
        <th>Score</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""

    html_path = export_dir / "dashboard.html"
    io.write_text(html_path, html_content)
    return html_path


def generate_marketing_exports(
    *, outdir: Path, limit: int = 1000, dataset_path: Optional[Path] = None
) -> dict:
    """Generate marketing exports and return a summary payload."""

    dataset = _load_dataset(outdir, dataset_path)
    dataset = dataset.copy()

    email_valid = _compute_email_valid(dataset)
    phone_valid = _compute_phone_valid(dataset)

    dataset["email_valid"] = email_valid
    dataset["telephone_valid"] = phone_valid

    dataset = _ensure_score(dataset)

    contactable_mask = email_valid | phone_valid
    contactable_df = dataset.loc[contactable_mask].copy()
    contactable_df = _deduplicate(contactable_df)

    contactable_df = contactable_df.sort_values(
        by=["score_business", "email_valid", "telephone_valid"],
        ascending=[False, False, False],
    )

    top_df = contactable_df.head(limit)

    export_df = _prepare_export_frame(top_df)
    export_dir = io.ensure_dir(outdir / "marketing_exports")

    csv_path = export_dir / f"top_contactables_{limit}.csv"
    parquet_path = export_dir / f"top_contactables_{limit}.parquet"
    export_df.to_csv(csv_path, index=False, encoding="utf-8")
    export_df.to_parquet(parquet_path, index=False)

    kpis = _compute_kpis(dataset, email_valid, phone_valid)
    dashboard_sample = export_df.head(min(len(export_df), 20))
    dashboard_path = _render_dashboard_html(export_dir, kpis, dashboard_sample)

    return {
        "status": "OK",
        "total_leads": kpis.total_leads,
        "contactable_leads": kpis.contactable_leads,
        "valid_email_rate": kpis.valid_email_rate,
        "duplicate_rate": kpis.duplicate_rate,
        "duplicate_key": kpis.duplicate_key,
        "top_contactables_csv": str(csv_path),
        "top_contactables_parquet": str(parquet_path),
        "dashboard_html": str(dashboard_path),
    }


__all__ = ["generate_marketing_exports", "MarketingKpis"]
