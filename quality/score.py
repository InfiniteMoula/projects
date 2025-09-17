
# FILE: quality/score.py
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _calculate_contactability(df: pd.DataFrame) -> pd.Series:
    """Calculate contactability score based on available contact information."""
    score = pd.Series(0.0, index=df.index, dtype="float64")
    
    # Score components (each worth up to 0.25, total = 1.0)
    # 1. Email availability and plausibility
    if "best_email" in df.columns:
        # Handle PyArrow nullable types by converting to string first
        email_series = df["best_email"].astype(str).fillna("").str.strip()
        email_valid = email_series.str.len() > 0
        if "email_plausible" in df.columns:
            email_score = pd.to_numeric(df["email_plausible"], errors="coerce").fillna(0.0)
        else:
            # Basic email format check if email_plausible not available
            email_score = email_series.str.contains(r"@.*\.", na=False).astype(float)
        score += email_valid.astype(float) * email_score * 0.35
    
    # 2. Phone number availability and validity
    if "telephone_norm" in df.columns:
        phone_series = df["telephone_norm"].astype(str).fillna("").str.strip()
        phone_valid = phone_series.str.len() > 0
        score += phone_valid.astype(float) * 0.25
    
    # 3. Website/domain availability
    if "domain_root" in df.columns:
        domain_series = df["domain_root"].astype(str).fillna("").str.strip()
        domain_valid = domain_series.str.len() > 0
        if "domain_valid" in df.columns:
            domain_score = pd.to_numeric(df["domain_valid"], errors="coerce").fillna(0.0)
        else:
            domain_score = 1.0  # Assume valid if present
        score += domain_valid.astype(float) * domain_score * 0.25
    
    # 4. Address completeness
    if "adresse_complete" in df.columns:
        address_series = df["adresse_complete"].astype(str).fillna("").str.strip()
        address_valid = address_series.str.len() > 10  # At least some address info
        score += address_valid.astype(float) * 0.15
    
    return score.clip(0.0, 1.0)


def _calculate_completeness(df: pd.DataFrame) -> pd.Series:
    """Calculate completeness score based on the presence of key business data."""
    total_fields = 0
    completed_fields = pd.Series(0, index=df.index, dtype="float64")
    
    # Key business fields to check
    key_fields = [
        "siren", "denomination", "raison_sociale", "naf_code", 
        "adresse_complete", "code_postal", "ville", "effectif"
    ]
    
    for field in key_fields:
        if field in df.columns:
            total_fields += 1
            field_complete = df[field].notna() & (df[field].astype(str).str.strip() != "")
            completed_fields += field_complete.astype(float)
    
    if total_fields == 0:
        return pd.Series(0.0, index=df.index, dtype="float64")
    
    return (completed_fields / total_fields).clip(0.0, 1.0)


def _calculate_unicity(df: pd.DataFrame) -> pd.Series:
    """Calculate unicity score based on duplicate detection across key identifiers."""
    score = pd.Series(1.0, index=df.index, dtype="float64")
    
    # Check for duplicates across multiple key fields
    dedup_fields = ["siren", "domain_root", "best_email", "telephone_norm"]
    available_fields = [f for f in dedup_fields if f in df.columns]
    
    if not available_fields:
        return score
    
    # For each available field, reduce score if duplicates are found
    for field in available_fields:
        field_data = df[field].astype(str).fillna("").str.strip()
        # Only consider non-empty values
        non_empty_mask = field_data.str.len() > 0
        
        if non_empty_mask.any():
            # Find duplicates among non-empty values
            duplicated = field_data.duplicated(keep=False) & non_empty_mask
            # Reduce score for duplicated entries
            score = score - (duplicated.astype(float) * 0.25)
    
    return score.clip(0.0, 1.0)


def _calculate_freshness(df: pd.DataFrame) -> pd.Series:
    """Calculate freshness score based on data recency and update timestamps."""
    score = pd.Series(0.8, index=df.index, dtype="float64")  # Default reasonable freshness
    
    # Check for date fields that indicate data freshness
    date_fields = ["date_creation", "date_maj", "last_updated"]
    
    for field in date_fields:
        if field in df.columns:
            try:
                # Try to parse dates and calculate age
                dates = pd.to_datetime(df[field], errors="coerce")
                current_date = pd.Timestamp.now()
                age_days = (current_date - dates).dt.days
                
                # Score based on age: newer data gets higher score
                # 100% for data < 1 year, decreasing to 50% for data > 3 years
                date_score = 1.0 - (age_days / (3 * 365)).clip(0, 0.5)
                
                # Use the best (most recent) date found
                score = score.combine(date_score.fillna(score), max)
                break  # Use first valid date field found
            except:
                continue
    
    return score.clip(0.0, 1.0)


def run(cfg: dict, ctx: dict) -> dict:
    """
    Calcule un score agr?g? robuste (sans NaN) et ?crit :
      - quality_score.parquet (une colonne score_quality)
      - quality_summary.json  (m?triques agr?g?es)
    Source pr?f?r?e : deduped.parquet (sinon enriched_email.parquet, sinon enriched_domain.parquet)
    Les colonnes manquantes sont calcul?es ? partir des donn?es disponibles.
    """

    t0 = time.time()
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))

    candidates = [
        outdir / "enriched_email.parquet",
        outdir / "enriched_domain.parquet", 
        outdir / "normalized.parquet",  # Prioritize normalized data since deduplication is removed
        outdir / "deduped.parquet",     # Keep as fallback in case it exists from previous runs
    ]
    src = next((p for p in candidates if p.exists()), None)
    if not src:
        return {"status": "WARN", "error": "no input parquet for scoring"}

    df = pq.read_table(src).to_pandas(types_mapper=pd.ArrowDtype)

    # Calculate quality metrics based on actual data
    df["contactability"] = _calculate_contactability(df)
    df["completeness"] = _calculate_completeness(df)
    df["unicity"] = _calculate_unicity(df)
    df["freshness"] = _calculate_freshness(df)

    # Ensure all quality metrics are numeric and handle any remaining NaN values
    needed = ["contactability", "unicity", "completeness", "freshness"]
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    weights = (cfg.get("scoring", {}) or {}).get("weights", {})
    w_contact = float(weights.get("contactability", 50))
    w_unicity = float(weights.get("unicity", 20))
    w_complete = float(weights.get("completeness", 20))
    w_fresh = float(weights.get("freshness", 10))
    w_sum = max(w_contact + w_unicity + w_complete + w_fresh, 1.0)

    df["score_quality"] = (
        df["contactability"] * w_contact +
        df["unicity"] * w_unicity +
        df["completeness"] * w_complete +
        df["freshness"] * w_fresh
    ) / w_sum
    df["score_quality"] = pd.to_numeric(df["score_quality"], errors="coerce").fillna(0.0)

    out_parquet = outdir / "quality_score.parquet"
    out_json = outdir / "quality_summary.json"

    pq.write_table(pa.Table.from_pandas(df[["score_quality"]], preserve_index=False), out_parquet, compression="snappy")

    summary = {
        "rows": int(len(df)),
        "score_mean": float(df["score_quality"].mean(skipna=True) or 0.0),
        "score_p50": float(df["score_quality"].quantile(0.50, interpolation="linear")),
        "score_p90": float(df["score_quality"].quantile(0.90, interpolation="linear")),
        "duration_s": round(time.time() - t0, 3),
    }
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return {
        "status": "OK",
        "files": [str(out_parquet), str(out_json)],
        **summary,
    }
