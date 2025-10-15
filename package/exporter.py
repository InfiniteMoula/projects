"""Export enriched dataset by merging normalized base records with contacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import pandas as pd

from quality.validation import FieldValidation, validate_email, validate_linkedin_url, validate_site_web, validate_telephone
from utils import io


def _normalize_siren(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = "".join(ch for ch in str(value) if ch.isdigit())
    if not text:
        return None
    return text[:9]


def _normalize_domain(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def _ensure_column(df: pd.DataFrame, target: str, aliases: Iterable[str]) -> None:
    if target in df.columns:
        return
    for alias in aliases:
        if alias in df.columns:
            df[target] = df[alias]
            return
    df[target] = pd.NA


def _load_contacts(outdir: Path) -> pd.DataFrame:
    contacts_dir = outdir / "contacts"
    candidates = [
        contacts_dir / "contacts_clean.parquet",
        contacts_dir / "contacts.parquet",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_parquet(path)
    return pd.DataFrame()


def _prepare_contacts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    contacts = df.copy()
    contacts["siren_key"] = contacts.get("siren").map(_normalize_siren)
    contacts["domain_key"] = contacts.get("domain", "").map(_normalize_domain)
    contacts["domain_key"] = contacts["domain_key"].fillna("")
    if "domain" in contacts.columns:
        contacts = contacts.rename(columns={"domain": "contact_domain"})
    return contacts


def _prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    _ensure_column(base, "domain", ["domain", "top_domain", "website", "site"])
    base["siren_key"] = base.get("siren").map(_normalize_siren)
    domain_series = base["domain"].map(_normalize_domain)
    base["domain_key"] = domain_series.fillna("")
    _ensure_column(base, "name", ["name", "denomination", "denomination_usuelle"])
    _ensure_column(base, "naf", ["naf", "naf_code", "ape"])
    _ensure_column(base, "region", ["region", "region_name", "region_label"])
    return base


def _merge_contacts(base_df: pd.DataFrame, contacts_df: pd.DataFrame) -> pd.DataFrame:
    if contacts_df.empty:
        result = base_df.copy()
        _ensure_column(result, "emails", [])
        _ensure_column(result, "phones", [])
        return result

    contacts = contacts_df.copy()
    contact_cols = [col for col in contacts.columns if col != "domain_key"]
    siren_contacts = (
        contacts[contacts["siren_key"].notna()]
        .drop_duplicates("siren_key", keep="first")
        .loc[:, contact_cols]
    )

    merged = base_df.merge(
        siren_contacts,
        how="left",
        on="siren_key",
        suffixes=("", "_contact"),
    )

    if "contact_domain" in merged.columns:
        merged["domain"] = merged["domain"].where(merged["domain"].notna(), merged["contact_domain"])

    domain_contacts = contacts[contacts["siren_key"].isna() & contacts["domain_key"].ne("")]
    if not domain_contacts.empty:
        domain_contacts = domain_contacts.drop_duplicates("domain_key", keep="first")
        domain_lookup = domain_contacts.set_index("domain_key")
        for column in domain_lookup.columns:
            if column in {"siren", "siren_key", "domain_key"}:
                continue
            fallback_series = merged["domain_key"].map(domain_lookup[column])
            if column == "contact_domain":
                merged["domain"] = merged["domain"].combine_first(fallback_series)
            else:
                if column not in merged.columns:
                    merged[column] = fallback_series
                else:
                    merged[column] = merged[column].combine_first(fallback_series)

    _ensure_column(merged, "emails", [])
    _ensure_column(merged, "phones", [])

    merged = merged.drop(columns=[col for col in ["siren_key", "domain_key", "contact_domain"] if col in merged.columns])
    return merged


def _compute_contact_validation_stats(df: pd.DataFrame) -> Dict[str, dict]:
    """Annotate dataframe with validation flags and return aggregated statistics."""
    stats: Dict[str, dict] = {}
    field_specs: Tuple[Tuple[str, Callable[[str], FieldValidation], str, Optional[str]], ...] = (
        ("site_web", validate_site_web, "site_web_valid", "site_web_normalized"),
        (
            "email",
            lambda value: validate_email(value, check_mx=False),
            "email_valid",
            "email_normalized",
        ),
        ("telephone", validate_telephone, "telephone_valid", "telephone_e164"),
        ("linkedin_url", validate_linkedin_url, "linkedin_url_valid", "linkedin_url_normalized"),
    )

    total_rows = len(df)
    index = df.index

    for column, validator, flag_name, normalized_name in field_specs:
        present = 0
        valid = 0
        flags = []
        normalized_values = []
        if column not in df.columns:
            df[flag_name] = pd.Series([pd.NA] * total_rows, index=index, dtype="boolean")
            if normalized_name:
                df[normalized_name] = pd.Series([pd.NA] * total_rows, index=index, dtype="string")
            stats[column] = {
                "present": 0,
                "valid": 0,
                "invalid": 0,
                "missing": total_rows,
                "valid_rate": 0.0,
            }
            continue

        series = df[column].astype("string").fillna("").str.strip()
        for value in series:
            if not value:
                flags.append(pd.NA)
                normalized_values.append(pd.NA)
                continue
            present += 1
            outcome = validator(value)
            is_valid = bool(outcome.is_valid)
            flags.append(is_valid)
            normalized_values.append(outcome.normalized if outcome.normalized is not None else value)
            if is_valid:
                valid += 1

        valid_series = pd.Series(flags, index=index, dtype="boolean")
        df[flag_name] = valid_series
        if normalized_name:
            df[normalized_name] = pd.Series(normalized_values, index=index, dtype="string")

        invalid = present - valid
        missing = total_rows - present
        stats[column] = {
            "present": present,
            "valid": valid,
            "invalid": invalid,
            "missing": missing,
            "valid_rate": round((valid / present * 100.0) if present else 0.0, 2),
        }

    return stats


def _write_quality_reports(outdir: Path, stats: Dict[str, dict], total_rows: int) -> dict:
    """Persist contact validation report to CSV, JSON, and PDF when possible."""
    report_rows = []
    for field, payload in stats.items():
        row = {"field": field}
        row.update(payload)
        report_rows.append(row)

    report_df = pd.DataFrame(report_rows)
    csv_path = outdir / "quality_contacts_report.csv"
    json_path = outdir / "quality_contacts_report.json"
    pdf_path: Optional[Path] = outdir / "quality_contacts_report.pdf"

    report_df.to_csv(csv_path, index=False, encoding="utf-8")
    total_present = int(sum(item["present"] for item in stats.values()))
    json_payload = {
        "fields": stats,
        "total_present": total_present,
        "total_rows": total_rows,
        "generated_at": io.now_iso(),
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    html_content = [
        "<!DOCTYPE html>",
        "<html lang='fr'>",
        "<head><meta charset='utf-8'><title>Rapport qualité contacts</title>",
        "<style>body{font-family:Arial, sans-serif;margin:24px;color:#1b1f23;}"
        "table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #d0d7de;padding:8px;text-align:left;}"
        "th{background:#f6f8fa;font-weight:600;}</style>",
        "</head><body>",
        "<h1>Rapport qualité des contacts</h1>",
        "<table>",
        "<tr><th>Champ</th><th>Présents</th><th>Valides</th><th>Invalides</th><th>Manquants</th><th>Taux valide (%)</th></tr>",
    ]

    for row in report_rows:
        html_content.append(
            "<tr>"
            f"<td>{row['field']}</td>"
            f"<td>{row['present']}</td>"
            f"<td>{row['valid']}</td>"
            f"<td>{row['invalid']}</td>"
            f"<td>{row['missing']}</td>"
            f"<td>{row['valid_rate']}</td>"
            "</tr>"
        )

    html_content.extend(["</table>", "</body></html>"])
    html = "\n".join(html_content)
    html_path = outdir / "quality_contacts_report.html"
    html_path.write_text(html, encoding="utf-8")

    try:
        from weasyprint import HTML  # type: ignore

        HTML(string=html).write_pdf(str(pdf_path))
    except Exception:
        pdf_path = None

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "html": str(html_path),
        "pdf": str(pdf_path) if pdf_path else None,
    }


def run(cfg: dict, ctx: dict) -> dict:
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    normalized_path = outdir / "normalized.parquet"

    if not normalized_path.exists():
        raise FileNotFoundError(f"normalized dataset not found at {normalized_path}")

    base_df = pd.read_parquet(normalized_path)
    base_prepared = _prepare_base(base_df)
    contacts_df = _prepare_contacts(_load_contacts(outdir))
    merged_df = _merge_contacts(base_prepared, contacts_df)

    for column in (
        "siren",
        "name",
        "naf",
        "region",
        "domain",
        "emails",
        "phones",
        "site_web",
        "email",
        "telephone",
        "linkedin_url",
    ):
        if column not in merged_df.columns:
            merged_df[column] = pd.NA

    records = len(merged_df)
    validation_stats = _compute_contact_validation_stats(merged_df)
    report_paths = _write_quality_reports(outdir, validation_stats, records)
    csv_path = outdir / "dataset_enriched.csv"
    parquet_path = outdir / "dataset_enriched.parquet"

    merged_df.to_csv(csv_path, index=False, encoding="utf-8")
    merged_df.to_parquet(parquet_path, index=False)

    return {
        "status": "OK",
        "records": records,
        "csv": str(csv_path),
        "parquet": str(parquet_path),
        "quality_report_csv": report_paths["csv"],
        "quality_report_json": report_paths["json"],
        "quality_report_html": report_paths["html"],
        "quality_report_pdf": report_paths["pdf"],
        "contact_validation_stats": validation_stats,
    }


__all__ = ["run"]
