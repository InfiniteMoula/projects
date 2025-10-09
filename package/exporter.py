"""Export enriched dataset by merging normalized base records with contacts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

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
    base["siren_key"] = base.get("siren").map(_normalize_siren)
    base["domain_key"] = base.get("domain", "").map(_normalize_domain)
    base["domain_key"] = base["domain_key"].fillna("")
    _ensure_column(base, "name", ["name", "denomination", "denomination_usuelle"])
    _ensure_column(base, "naf", ["naf", "naf_code", "ape"])
    _ensure_column(base, "region", ["region", "region_name", "region_label"])
    _ensure_column(base, "domain", ["domain", "top_domain", "website", "site"])
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


def run(cfg: dict, ctx: dict) -> dict:
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    normalized_path = outdir / "normalized.parquet"

    if not normalized_path.exists():
        raise FileNotFoundError(f"normalized dataset not found at {normalized_path}")

    base_df = pd.read_parquet(normalized_path)
    base_prepared = _prepare_base(base_df)
    contacts_df = _prepare_contacts(_load_contacts(outdir))
    merged_df = _merge_contacts(base_prepared, contacts_df)

    for column in ("siren", "name", "naf", "region", "domain", "emails", "phones"):
        if column not in merged_df.columns:
            merged_df[column] = pd.NA

    records = len(merged_df)
    csv_path = outdir / "dataset_enriched.csv"
    parquet_path = outdir / "dataset_enriched.parquet"

    merged_df.to_csv(csv_path, index=False, encoding="utf-8")
    merged_df.to_parquet(parquet_path, index=False)

    return {
        "status": "OK",
        "records": records,
        "csv": str(csv_path),
        "parquet": str(parquet_path),
    }


__all__ = ["run"]
