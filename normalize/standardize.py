
# FILE: normalize/standardize.py
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa

from utils import io
from utils.parquet import ArrowCsvWriter, ParquetBatchWriter, iter_batches

# --- config -------------------------------------------------------------
TEL_RE = re.compile(r"\D+")

# Placeholder values that should be treated as missing data
PLACEHOLDER_VALUES = {
    "TELEPHONE NON RENSEIGNE",
    "ADRESSE NON RENSEIGNEE", 
    "DENOMINATION NON RENSEIGNEE"
}

# on inclut les variantes CamelCase + snake_case courantes
DEFAULT_USECOLS = [
    "siren","siret","nic",
    "denominationUniteLegale","denominationusuelleetablissement","denominationUsuelleEtablissement",
    "enseigne1etablissement","enseigne1Etablissement",
    "libellecommuneetablissement","libelleCommuneEtablissement",
    "codepostaletablissement","codePostalEtablissement",
    "adresseetablissement","adresseEtablissement",
    "numeroVoieEtablissement","typeVoieEtablissement","libelleVoieEtablissement",
    "complementAdresseEtablissement",
    "activitePrincipaleEtablissement","activiteprincipaleetablissement",
    "activitePrincipaleUniteLegale","activiteprincipaleunitelegale",
    "dateCreationEtablissement","datecreationetablissement",
    "nomunitelegale","nomUniteLegale","prenomsunitelegale","prenomsUniteLegale",
    "telephone","email","siteweb",
    "etatAdministratifEtablissement","etatadministratifetablissement",
]

ARROW_OUT_SCHEMA = pa.schema([
    ("siren", pa.string()),
    ("siret", pa.string()),
    ("raison_sociale", pa.string()),
    ("enseigne", pa.string()),
    ("commune", pa.string()),
    ("cp", pa.string()),
    ("adresse", pa.string()),
    ("naf", pa.string()),
    ("date_creation", pa.string()),
    ("telephone_norm", pa.string()),
    ("email", pa.string()),
    ("siteweb", pa.string()),
    ("nom", pa.string()),
    ("prenom", pa.string()),
])

# --- helpers ------------------------------------------------------------
def _is_placeholder(s: pd.Series) -> pd.Series:
    """Check if values in a Series are placeholder values that should be treated as missing data."""
    if s is None:
        return pd.Series(False, dtype=bool)
    return s.astype("string").isin(PLACEHOLDER_VALUES)


def _to_str(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(pd.NA, dtype="string")
    if str(s.dtype) != "string":
        return s.astype("string", copy=False)
    return s


def _fr_tel_norm(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(pd.NA, dtype="string")
    x = s.fillna("").astype("string", copy=False).str.replace(TEL_RE, "", regex=True)

    def _fmt(v: str) -> str:
        if not v:
            return ""
        if len(v) == 10 and v.startswith("0"):
            return "+33" + v[1:]
        if len(v) == 9:
            return "+33" + v
        return v

    x = x.map(_fmt).replace({"": pd.NA})
    return x.astype("string")


def _pick_first(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for name in names:
        if name in df.columns:
            return df[name]
    return None


# --- main ---------------------------------------------------------------
def run(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    input_path = ctx.get("input_path") or ctx.get("input")
    input_path = Path(input_path) if input_path else None
    if input_path is None or not input_path.exists():
        return {"status": "FAIL", "error": f"Input parquet not found: {input_path}"}

    job = ctx.get("job", {}) or {}
    filters = (job.get("filters") or {})
    # Keep original case and just clean spaces/dots for better matching
    naf_include_raw = [x.replace(".", "").replace(" ", "") for x in (filters.get("naf_include") or [])]
    naf_prefixes = tuple(filter(None, naf_include_raw))
    active_only = bool(filters.get("active_only", False))

    usecols = job.get("standardize_usecols", DEFAULT_USECOLS)
    batch_rows = int(job.get("standardize_batch_rows", 200_000))

    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    out_parquet = outdir / "normalized.parquet"
    out_csv = outdir / "normalized.csv"

    for target in (out_parquet, out_csv):
        if target.exists():
            try:
                target.unlink()
            except OSError:
                pass

    total = 0
    raw_rows_total = 0
    filtered_rows_total = 0
    batches = 0
    logger = ctx.get("logger")

    try:
        with ParquetBatchWriter(out_parquet, schema=ARROW_OUT_SCHEMA) as pq_writer, ArrowCsvWriter(out_csv) as csv_writer:
            for pdf in iter_batches(input_path, columns=usecols, batch_size=batch_rows):
                batches += 1
                raw_rows = len(pdf)
                raw_rows_total += raw_rows
                if raw_rows == 0:
                    continue

                object_cols = [col for col in pdf.columns if str(pdf[col].dtype) == "object"]
                if object_cols:
                    pdf[object_cols] = pdf[object_cols].astype("string", copy=False)

                naf_col = next(
                    (cand for cand in [
                        "activitePrincipaleEtablissement","activiteprincipaleetablissement",
                        "activitePrincipaleUniteLegale","activiteprincipaleunitelegale"
                    ] if cand in pdf.columns),
                    None,
                )

                if naf_prefixes and naf_col:
                    # Improved NAF filtering: handle subcategories more inclusively for business activity codes
                    naf_norm = _to_str(pdf[naf_col]).str.replace(r"[\s\.]", "", regex=True).str.upper()
                    combined_mask = pd.Series([False] * len(pdf))
                    
                    for prefix in naf_prefixes:
                        # Clean the prefix the same way as the data
                        prefix_clean = prefix.replace(".", "").replace(" ", "").upper()
                        
                        # Apply smart matching for 4+ digit codes ending with letters (common business subcategories)
                        # This helps with codes like 6920Z (accounting) to also match 6920A, 6920B, etc.
                        # But we want to be more restrictive for agricultural/forestry codes (01.XXZ, 02.XXZ)
                        if (len(prefix_clean) >= 4 and 
                            prefix_clean and prefix_clean[-1].isalpha() and
                            prefix_clean[:-1].isdigit() and 
                            not prefix_clean.startswith(('01', '02', '03'))):  # Exclude agriculture/forestry
                            
                            base_code = prefix_clean[:-1]  # Remove letter suffix
                            # Match either the specific code or the base category
                            mask = (naf_norm.fillna("").str.startswith(base_code) | 
                                   naf_norm.fillna("").str.startswith(prefix_clean))
                        else:
                            # Standard prefix matching for other codes (numeric or short codes)
                            mask = naf_norm.fillna("").str.startswith(prefix_clean)
                        
                        combined_mask = combined_mask | mask
                    
                    pdf = pdf[combined_mask.fillna(False)]

                if active_only:
                    state_col = next((c for c in [
                        "etatAdministratifEtablissement","etatadministratifetablissement"
                    ] if c in pdf.columns), None)
                    if state_col:
                        pdf = pdf[_to_str(pdf[state_col]).eq("A")]

                filtered_rows = len(pdf)
                filtered_rows_total += filtered_rows
                if filtered_rows == 0:
                    continue

                # Extract raison_sociale with fallback logic
                raison_sociale = _to_str(_pick_first(pdf, ["denominationUniteLegale","denominationunitelegale"]))
                # Fallback to denominationUsuelleEtablissement when denominationUniteLegale is empty/missing
                fallback_raison_sociale = _to_str(_pick_first(pdf, ["denominationUsuelleEtablissement","denominationusuelleetablissement"]))
                raison_sociale = raison_sociale.fillna("").where(
                    raison_sociale.fillna("").str.len() > 0,
                    fallback_raison_sociale
                )
                # Convert empty strings to null for better data quality
                raison_sociale = raison_sociale.replace("", pd.NA)

                # Extract enseigne with fallback logic
                enseigne = _to_str(_pick_first(pdf, ["enseigne1Etablissement","enseigne1etablissement"]))
                # Fallback to raison_sociale if enseigne is missing
                enseigne = enseigne.fillna("").where(
                    enseigne.fillna("").str.len() > 0,
                    raison_sociale
                )
                # Convert empty strings to null for better data quality
                enseigne = enseigne.replace("", pd.NA)

                # Extract adresse - preserve missing values as null instead of placeholder
                adresse = _to_str(_pick_first(pdf, ["adresseEtablissement","adresseetablissement"]))
                # Convert empty strings to null values for better data quality
                adresse = adresse.replace("", pd.NA)

                # Extract telephone - preserve missing values as null instead of placeholder
                telephone_norm = _fr_tel_norm(_pick_first(pdf, ["telephone"]))

                res = pd.DataFrame({
                    "siren": _to_str(_pick_first(pdf, ["siren"])),
                    "siret": _to_str(_pick_first(pdf, ["siret"])),
                    "raison_sociale": raison_sociale,
                    "enseigne": enseigne,
                    "commune": _to_str(_pick_first(pdf, ["libelleCommuneEtablissement","libellecommuneetablissement"])),
                    "cp": _to_str(_pick_first(pdf, ["codePostalEtablissement","codepostaletablissement"])),
                    "adresse": adresse,
                    "naf": _to_str(_pick_first(pdf, [
                        "activitePrincipaleEtablissement","activiteprincipaleetablissement",
                        "activitePrincipaleUniteLegale","activiteprincipaleunitelegale"
                    ])).str.replace(r"\s", "", regex=True),
                    "date_creation": _to_str(_pick_first(pdf, ["dateCreationEtablissement","datecreationetablissement"])),
                    "telephone_norm": telephone_norm,
                    "email": _to_str(_pick_first(pdf, ["email"])),
                    "siteweb": _to_str(_pick_first(pdf, ["siteweb"])),
                    "nom": _to_str(_pick_first(pdf, ["nomUniteLegale","nomunitelegale"])),
                    "prenom": _to_str(_pick_first(pdf, ["prenomsUniteLegale","prenomsunitelegale"])),
                })

                # Convert empty strings to null values for better data quality in optional fields
                optional_fields = ["commune", "email", "siteweb", "nom", "prenom", "date_creation"]
                for field in optional_fields:
                    res[field] = res[field].replace("", pd.NA)

                res["cp"] = res["cp"].str.extract(r"(\d{5})", expand=False).astype("string")
                res["naf"] = res["naf"].astype("string")

                rows_written = len(res)
                total += rows_written
                if rows_written == 0:
                    continue

                table = pa.Table.from_pandas(res, preserve_index=False).cast(ARROW_OUT_SCHEMA)
                pq_writer.write_table(table)
                csv_writer.write_table(table)

        elapsed = time.time() - t0
        duration = round(elapsed, 3)
        rows_per_s = total / elapsed if elapsed > 0 else 0.0
        drop_pct = ((raw_rows_total - total) / raw_rows_total * 100) if raw_rows_total else 0.0

        kpi_targets = (job.get("kpi_targets") or {})
        kpi_evaluations: list[dict[str, object]] = []
        min_lines_target_raw = kpi_targets.get("min_lines_per_s")
        min_lines_target = float(min_lines_target_raw) if min_lines_target_raw is not None else None
        if min_lines_target is not None:
            kpi_evaluations.append({
                "name": "lines_per_s",
                "actual": rows_per_s,
                "target": min_lines_target,
                "met": rows_per_s >= min_lines_target,
            })

        summary = {
            "status": "OK",
            "step": "normalize.standardize",
            "batches": batches,
            "rows_raw": raw_rows_total,
            "rows_after_filters": filtered_rows_total,
            "rows_written": total,
            "rows_dropped": max(raw_rows_total - total, 0),
            "drop_pct": round(drop_pct, 2),
            "duration_s": duration,
            "rows_per_s": round(rows_per_s, 3),
            "filters": {
                "active_only": active_only,
                "naf_prefixes": list(naf_prefixes),
                "usecols_count": len(usecols) if hasattr(usecols, "__len__") else None,
            },
            "files": {
                "parquet": str(out_parquet),
                "csv": str(out_csv),
            },
        }
        if kpi_evaluations:
            summary["kpi_evaluations"] = kpi_evaluations
            summary["kpi_status"] = "MET" if all(item["met"] for item in kpi_evaluations) else "WARN"
            summary["kpi_targets"] = {"min_lines_per_s": min_lines_target}

        reports_dir = outdir / "reports"
        report_path = io.write_json(reports_dir / "standardize_summary.json", summary)
        summary["report_path"] = str(report_path)

        log_path = ctx.get("logs")
        if log_path:
            io.log_json(log_path, {
                "step": "normalize.standardize",
                "event": "summary",
                "status": summary.get("kpi_status", "OK"),
                "rows": total,
                "rows_per_s": round(rows_per_s, 3),
                "drop_pct": round(drop_pct, 2),
                "report_path": str(report_path),
            })

        if logger:
            kpi_phrase = ""
            if kpi_evaluations:
                details = "; ".join(
                    f"{item['name']} {'OK' if item['met'] else 'WARN'} ({item['actual']:.2f}/{item['target']:.2f})"
                    for item in kpi_evaluations
                )
                kpi_phrase = f" | KPI {details}"
            logger.info(
                "normalize.standardize summary | rows=%d | drop_pct=%.1f%% | rows_per_s=%.2f%s",
                total,
                drop_pct,
                rows_per_s,
                kpi_phrase,
            )

        files = [str(out_parquet), str(out_csv), str(report_path)]
        return {
            "status": "OK",
            "files": files,
            "rows": total,
            "duration_s": duration,
            "rows_per_s": round(rows_per_s, 3),
            "summary": summary,
            "report_path": str(report_path),
        }

    except Exception as exc:
        if logger:
            logger.exception("standardize failed: %s", exc)
        return {"status": "FAIL", "error": str(exc), "duration_s": round(time.time() - t0, 3)}

