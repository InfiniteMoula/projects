
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

# Column grouping patterns for merging similar columns
COLUMN_GROUPS = {
    # Core identifiers - keep separate as they are unique
    'siren': ['siren', 'SIREN'],
    'siret': ['siret', 'SIRET'], 
    'nic': ['nic', 'NIC'],
    
    # Company names and denominations - merge similar variations
    'denomination': [
        'denominationUniteLegale', 'denominationunitelegale', 'DENOMINATIONUNITELEGALE',
        'denomination', 'DENOMINATION', 'company_name', 'COMPANY_NAME',
        'raison_sociale', 'RAISON_SOCIALE'
    ],
    'denomination_usuelle': [
        'denominationUsuelleEtablissement', 'denominationusuelleetablissement', 'DENOMINATIONUSUELLEETABLISSEMENT'
    ],
    'enseigne': [
        'enseigne1Etablissement', 'enseigne1etablissement', 'ENSEIGNE1ETABLISSEMENT',
        'enseigne', 'ENSEIGNE', 'nom_commercial', 'NOM_COMMERCIAL'
    ],
    
    # Geographic information
    'commune': [
        'libelleCommuneEtablissement', 'libellecommuneetablissement', 'LIBELLECOMMUNEETABLISSEMENT',
        'commune', 'COMMUNE', 'ville', 'VILLE'
    ],
    'code_postal': [
        'codePostalEtablissement', 'codepostaletablissement', 'CODEPOSTALETABLISSEMENT',
        'code_postal', 'CODE_POSTAL', 'cp', 'CP'
    ],
    
    # Address information - merge all address variations
    'adresse': [
        'adresseEtablissement', 'adresseetablissement', 'ADRESSEETABLISSEMENT',
        'adresse', 'ADRESSE', 'address', 'ADDRESS',
        'adresse_complete', 'ADRESSE_COMPLETE', 'adresse_ligne_1', 'ADRESSE_LIGNE_1'
    ],
    'numero_voie': [
        'numeroVoieEtablissement', 'numerovoieetablissement', 'NUMEROVOIEETABLISSEMENT'
    ],
    'type_voie': [
        'typeVoieEtablissement', 'typevoieetablissement', 'TYPEVOIEETABLISSEMENT'
    ],
    'libelle_voie': [
        'libelleVoieEtablissement', 'libellevoieetablissement', 'LIBELLEVOIEETABLISSEMENT'
    ],
    'complement_adresse': [
        'complementAdresseEtablissement', 'complementadresseetablissement', 'COMPLEMENTADRESSEETABLISSEMENT'
    ],
    
    # Activity codes
    'naf': [
        'activitePrincipaleEtablissement', 'activiteprincipaleetablissement', 'ACTIVITEPRINCIPALEESTABLISSEMENT',
        'activitePrincipaleUniteLegale', 'activiteprincipaleunitelegale', 'ACTIVITEPRINCIPALEUNITELLEGALE',
        'naf', 'NAF', 'code_naf', 'CODE_NAF'
    ],
    
    # Dates
    'date_creation': [
        'dateCreationEtablissement', 'datecreationetablissement', 'DATECREATIONETABLISSEMENT',
        'date_creation', 'DATE_CREATION'
    ],
    
    # Personal names - merge all name variations
    'nom': [
        'nomUniteLegale', 'nomunitelegale', 'NOMUNITELEGALE',
        'nom', 'NOM', 'name', 'NAME', 'noms', 'NOMS'
    ],
    'prenom': [
        'prenomsUniteLegale', 'prenomsunitelegale', 'PRENOMSUNITELEGALE',
        'prenom', 'PRENOM', 'prenoms', 'PRENOMS'
    ],
    
    # Contact information - merge all variations
    'telephone': [
        'telephone', 'TELEPHONE', 'phone', 'PHONE', 'tel', 'TEL'
    ],
    'telephone_mobile': [
        'tel_mobile', 'TEL_MOBILE', 'mobile', 'MOBILE', 'portable', 'PORTABLE'
    ],
    'fax': [
        'fax', 'FAX', 'telecopie', 'TELECOPIE'
    ],
    'email': [
        'email', 'EMAIL', 'mail', 'MAIL', 'e_mail', 'E_MAIL'
    ],
    'website': [
        'siteweb', 'SITEWEB', 'site_web', 'SITE_WEB', 'website', 'WEBSITE', 'url', 'URL'
    ],
    
    # Administrative info
    'etat_administratif': [
        'etatAdministratifEtablissement', 'etatadministratifetablissement', 'ETATADMINISTRATIFETABLISSEMENT'
    ],
    'effectif': [
        'trancheEffectifsEtablissement', 'trancheeffectifsetablissement', 'TRANCHEEFFECTIFSETABLISSEMENT',
        'trancheEffectifsUniteLegale', 'trancheeffectifsunitelegale', 'TRANCHEEFFECTIFSUNITELEGALE',
        'effectif', 'EFFECTIF', 'effectif_salarie', 'EFFECTIF_SALARIE', 'tranche_effectif', 'TRANCHE_EFFECTIF'
    ],
    
    # Additional business information
    'secteur_activite': [
        'secteur_activite', 'SECTEUR_ACTIVITE', 'secteur', 'SECTEUR'
    ],
    'forme_juridique': [
        'forme_juridique', 'FORME_JURIDIQUE', 'forme', 'FORME'
    ],
    'capital_social': [
        'capital_social', 'CAPITAL_SOCIAL', 'capital', 'CAPITAL'
    ]
}

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


def _to_str_filtered(s: pd.Series | None) -> pd.Series:
    """Convert to string and filter out placeholder values."""
    if s is None:
        return pd.Series(pd.NA, dtype="string")
    
    str_series = _to_str(s)
    # Replace placeholder values with NA
    is_placeholder_mask = str_series.isin(PLACEHOLDER_VALUES)
    str_series = str_series.where(~is_placeholder_mask, pd.NA)
    
    # Replace empty strings with NA
    str_series = str_series.replace("", pd.NA)
    
    return str_series


def _fr_tel_norm(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(pd.NA, dtype="string")
    
    # First filter out placeholder values
    str_series = _to_str(s)
    is_placeholder_mask = str_series.isin(PLACEHOLDER_VALUES)
    str_series = str_series.where(~is_placeholder_mask, "")
    
    # Now normalize phone numbers
    x = str_series.fillna("").str.replace(TEL_RE, "", regex=True)

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
    """Pick the first available column from a list of names, supporting case-insensitive matching."""
    # First try exact match (for performance)
    for name in names:
        if name in df.columns:
            return df[name]
    
    # If no exact match, try case-insensitive matching
    df_columns_lower = {col.lower(): col for col in df.columns}
    for name in names:
        name_lower = name.lower()
        if name_lower in df_columns_lower:
            return df[df_columns_lower[name_lower]]
    
    return None


def _merge_columns(df: pd.DataFrame, column_names: list[str], prefer_non_empty: bool = True) -> pd.Series:
    """Merge multiple columns by combining their values, prioritizing non-empty values."""
    result = pd.Series(pd.NA, index=df.index, dtype="string")
    
    for col_name in column_names:
        if col_name in df.columns:
            col_data = _to_str_filtered(df[col_name])
            if prefer_non_empty:
                # Fill empty values in result with values from this column
                result = result.fillna(col_data)
            else:
                # Simple concatenation (could be used for other merge strategies)
                result = result.combine_first(col_data)
    
    return result


def _extract_all_columns(df: pd.DataFrame) -> dict:
    """Extract and merge all available columns based on column groups."""
    result = {}
    used_columns = set()
    
    # Process each column group
    for group_name, column_patterns in COLUMN_GROUPS.items():
        available_cols = [col for col in column_patterns if col in df.columns]
        if available_cols:
            # Mark these columns as used
            used_columns.update(available_cols)
            # Merge the columns
            result[group_name] = _merge_columns(df, available_cols)
    
    # Add any remaining columns that weren't matched to groups
    remaining_cols = set(df.columns) - used_columns
    for col in remaining_cols:
        # Clean column name for output (lowercase, replace spaces/special chars)
        clean_name = col.lower().replace(' ', '_').replace('-', '_').replace('.', '_')
        # Avoid name conflicts
        if clean_name in result:
            clean_name = f"{clean_name}_extra"
        result[clean_name] = _to_str_filtered(df[col])
    
    return result


def _create_dynamic_schema(columns: dict) -> pa.Schema:
    """Create a PyArrow schema based on the actual columns present."""
    fields = []
    for col_name in sorted(columns.keys()):
        fields.append((col_name, pa.string()))
    return pa.schema(fields)


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

    # Remove usecols restriction to extract all columns
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
    dynamic_schema = None

    try:
        # Initialize writers without fixed schema - will be created dynamically
        pq_writer = None
        csv_writer = None
        
        for pdf in iter_batches(input_path, columns=None, batch_size=batch_rows):  # Extract all columns
            batches += 1
            raw_rows = len(pdf)
            raw_rows_total += raw_rows
            if raw_rows == 0:
                continue

            object_cols = [col for col in pdf.columns if str(pdf[col].dtype) == "object"]
            if object_cols:
                pdf[object_cols] = pdf[object_cols].astype("string", copy=False)

            # Apply NAF filtering
            naf_col_series = _pick_first(pdf, [
                "activitePrincipaleEtablissement","activiteprincipaleetablissement","ACTIVITEPRINCIPALEESTABLISSEMENT",
                "activitePrincipaleUniteLegale","activiteprincipaleunitelegale","ACTIVITEPRINCIPALEUNITELLEGALE"
            ])

            if naf_prefixes and naf_col_series is not None:
                # Improved NAF filtering
                naf_norm = _to_str(naf_col_series).str.replace(r"[\s\.]", "", regex=True).str.upper()
                combined_mask = pd.Series([False] * len(pdf))
                
                for prefix in naf_prefixes:
                    # Clean the prefix the same way as the data
                    prefix_clean = prefix.replace(".", "").replace(" ", "").upper()
                    
                    # Apply smart matching for 4+ digit codes ending with letters
                    if (len(prefix_clean) >= 4 and 
                        prefix_clean and prefix_clean[-1].isalpha() and
                        prefix_clean[:-1].isdigit() and 
                        not prefix_clean.startswith(('01', '02', '03'))):  # Exclude agriculture/forestry
                        
                        base_code = prefix_clean[:-1]  # Remove letter suffix
                        # Match either the specific code or the base category
                        mask = (naf_norm.fillna("").str.startswith(base_code) | 
                               naf_norm.fillna("").str.startswith(prefix_clean))
                    else:
                        # Standard prefix matching for other codes
                        mask = naf_norm.fillna("").str.startswith(prefix_clean)
                    
                    combined_mask = combined_mask | mask
                
                pdf = pdf[combined_mask.fillna(False)]

            if active_only:
                state_series = _pick_first(pdf, [
                    "etatAdministratifEtablissement","etatadministratifetablissement","ETATADMINISTRATIFETABLISSEMENT"
                ])
                if state_series is not None:
                    pdf = pdf[_to_str(state_series).eq("A")]

            filtered_rows = len(pdf)
            filtered_rows_total += filtered_rows
            if filtered_rows == 0:
                continue

            # Extract all columns using the comprehensive approach
            extracted_data = _extract_all_columns(pdf)
            
            # Apply special processing for phone numbers
            if 'telephone' in extracted_data:
                extracted_data['telephone_norm'] = _fr_tel_norm(extracted_data['telephone'])
            if 'telephone_mobile' in extracted_data:
                extracted_data['telephone_mobile_norm'] = _fr_tel_norm(extracted_data['telephone_mobile'])
            if 'fax' in extracted_data:
                extracted_data['fax_norm'] = _fr_tel_norm(extracted_data['fax'])
            
            # Clean postal codes
            if 'code_postal' in extracted_data:
                extracted_data['code_postal'] = extracted_data['code_postal'].str.extract(r"(\d{5})", expand=False).astype("string")
                # Add legacy alias
                extracted_data['cp'] = extracted_data['code_postal']
            
            # Clean NAF codes
            if 'naf' in extracted_data:
                extracted_data['naf'] = extracted_data['naf'].str.replace(r"\s", "", regex=True).astype("string")
                # Add legacy alias
                extracted_data['naf_code'] = extracted_data['naf']
            
            # Add backward compatibility aliases
            if 'denomination' in extracted_data:
                extracted_data['raison_sociale'] = extracted_data['denomination']
            if 'commune' in extracted_data:
                extracted_data['ville'] = extracted_data['commune'] 
            if 'adresse' in extracted_data:
                extracted_data['adresse_complete'] = extracted_data['adresse']
            if 'website' in extracted_data:
                extracted_data['siteweb'] = extracted_data['website']

            # Create DataFrame with all extracted data
            res = pd.DataFrame(extracted_data)
            
            # Ensure we have at least the basic required columns for compatibility
            required_cols = ['siren', 'siret']
            for col in required_cols:
                if col not in res.columns:
                    res[col] = pd.Series(pd.NA, index=res.index, dtype="string")

            rows_written = len(res)
            total += rows_written
            if rows_written == 0:
                continue

            # Create or update dynamic schema
            if dynamic_schema is None:
                dynamic_schema = _create_dynamic_schema(extracted_data)
                pq_writer = ParquetBatchWriter(out_parquet, schema=dynamic_schema)
                csv_writer = ArrowCsvWriter(out_csv)

            # Ensure DataFrame columns are in the same order as schema
            schema_columns = [field.name for field in dynamic_schema]
            res = res.reindex(columns=schema_columns, fill_value=pd.NA)

            table = pa.Table.from_pandas(res, preserve_index=False).cast(dynamic_schema)
            pq_writer.write_table(table)
            csv_writer.write_table(table)

        # Close writers
        if pq_writer:
            pq_writer.close()
        if csv_writer:
            csv_writer.close()

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
                "comprehensive_extraction": True,  # Indicate new approach
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

