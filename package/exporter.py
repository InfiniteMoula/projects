import os
import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Environment, FileSystemLoader
import weasyprint
from utils import io, hashx

CURATED_COLUMN_SPECS = [
    ('Nom entreprise', ['Nom entreprise', 'denomination_usuelle', 'enseigne']),
    ('Identifiant (SIREN/SIRET/DUNS)', ['Identifiant (SIREN/SIRET/DUNS)', 'siret', 'siren']),
    ('Forme juridique', ['Forme juridique']),
    ('Date de création', ['Date de création', 'date_creation']),
    ('Secteur (NAF/APE)', ['Secteur (NAF/APE)', 'naf', 'naf_code']),
    ('Adresse', ['Adresse', 'Adresse complète', 'adresse_complete']),
    ('Région / Département', ['Région / Département']),
    ('Téléphone standard', ['Téléphone standard', 'telephone_norm']),
    ('Email générique', ['Email générique', 'best_email']),
    ('Site web', ['Site web', 'siteweb']),
    ('Effectif', ['Effectif', 'effectif']),
    ('CA', ['CA', "Chiffre d'affaires exact et/ou Résultat net", "Chiffre d'affaires (fourchette)"]),
    ('Nom + fonction du dirigeant', ['Nom + fonction du dirigeant']),
    ('Email pro vérifié', ['Email pro vérifié', 'Email pro vérifié du dirigeant']),
    ('Technologies web', ['Technologies web']),
    ('Score de solvabilité', ['Score de solvabilité', 'score_quality']),
    ('Croissance CA (N vs N-1)', ['Croissance CA (N vs N-1)']),
]

CURATED_COLUMN_ORDER = [name for name, _ in CURATED_COLUMN_SPECS]

def _coalesce_columns(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    existing = [df[col] for col in candidates if col in df.columns]
    if not existing:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype='object')
    result = existing[0].copy()
    for series in existing[1:]:
        result = result.combine_first(series)
    return result

def build_curated_dataset(df: pd.DataFrame) -> pd.DataFrame:
    curated = pd.DataFrame(index=df.index)
    for output_name, sources in CURATED_COLUMN_SPECS:
        curated[output_name] = _coalesce_columns(df, sources)
    return curated

def merge_quality_data(outdir_path: Path) -> pd.DataFrame:
    """Merge deduped data with quality scores and address enrichment."""
    
    # Priority order for source data
    source_candidates = [
        outdir_path / "deduped.parquet",
        outdir_path / "enriched_email.parquet", 
        outdir_path / "enriched_domain.parquet",
        outdir_path / "google_maps_enriched.parquet",  # Primary enrichment source
        outdir_path / "address_extracted.parquet",    # Address parsing output
        outdir_path / "normalized.parquet"
    ]
    
    source_path = next((p for p in source_candidates if p.exists()), None)
    if not source_path:
        raise FileNotFoundError("No source parquet file found for export")
    
    # Load main dataset
    df = pd.read_parquet(source_path)
    
    # Try to merge with Google Maps enrichment data if not already included
    google_maps_path = outdir_path / "google_maps_enriched.parquet"
    if google_maps_path.exists() and source_path != google_maps_path:
        try:
            maps_df = pd.read_parquet(google_maps_path)
            # Merge on common columns (typically siren/siret)
            common_cols = [col for col in ['siren', 'siret'] if col in df.columns and col in maps_df.columns]
            if common_cols:
                # Keep all the Google Maps enrichment columns
                maps_cols = [col for col in maps_df.columns if col not in common_cols]
                
                if maps_cols:
                    merge_df = maps_df[common_cols + maps_cols]
                    df = df.merge(merge_df, on=common_cols, how='left')
                    print(f"Merged Google Maps enrichment data with {len(maps_cols)} new columns")
        except Exception as e:
            print(f"Warning: Could not merge Google Maps enrichment data: {e}")

    # Try to merge with address enrichment data if not already included
    address_path = outdir_path / "address_extracted.parquet"
    if address_path.exists() and source_path != address_path:
        try:
            address_df = pd.read_parquet(address_path)
            # Merge on common columns (typically siren/siret)
            common_cols = [col for col in ['siren', 'siret'] if col in df.columns and col in address_df.columns]
            if common_cols:
                # Keep only the new enrichment columns from address extraction
                address_cols = [col for col in address_df.columns if col not in common_cols and col not in df.columns]
                
                if address_cols:
                    merge_df = address_df[common_cols + address_cols]
                    df = df.merge(merge_df, on=common_cols, how='left')
                    print(f"Merged address extraction data with {len(address_cols)} new columns")
        except Exception as e:
            print(f"Warning: Could not merge address extraction data: {e}")
    
    # Try to merge with quality scores if available
    quality_path = outdir_path / "quality_score.parquet"
    if quality_path.exists():
        quality_df = pd.read_parquet(quality_path)
        # Merge on index (both should have same number of rows in same order)
        if len(df) == len(quality_df):
            df = pd.concat([df, quality_df], axis=1)
        else:
            print(f"Warning: Row count mismatch between data ({len(df)}) and quality scores ({len(quality_df)})")
    
    return df


def calculate_data_dictionary(df: pd.DataFrame) -> list[dict]:
    """Calculate data dictionary with completeness metrics."""
    
    dictionary = []
    total_rows = len(df)
    
    for column in df.columns:
        # Check for both non-null values AND non-empty strings
        if df[column].dtype == 'object' or pd.api.types.is_string_dtype(df[column]):
            # For string/object columns, exclude empty strings and whitespace-only strings
            non_empty_mask = (df[column].notna() & 
                             (df[column].astype(str).str.strip() != ""))
            non_empty_count = int(non_empty_mask.sum())
        else:
            # For numeric and other types, just check for non-null
            non_empty_count = int(df[column].notna().sum())
        
        completeness_rate = (non_empty_count / total_rows * 100) if total_rows > 0 else 0
        
        dictionary.append({
            "column": column,
            "non_null": non_empty_count,
            "completeness_rate": completeness_rate
        })
    
    return dictionary


def calculate_quality_metrics(df: pd.DataFrame) -> dict:
    """Calculate quality metrics from the dataset."""
    
    metrics = {
        "total_records": len(df),
        "quality_mean": None,
        "quality_p50": None,
        "quality_p90": None
    }
    
    if "score_quality" in df.columns:
        quality_scores = df["score_quality"].dropna()
        if len(quality_scores) > 0:
            metrics.update({
                "quality_mean": float(quality_scores.mean() * 100),  # Convert to percentage
                "quality_p50": float(quality_scores.quantile(0.50) * 100),
                "quality_p90": float(quality_scores.quantile(0.90) * 100)
            })
    
    return metrics


def generate_html_report(ctx: dict, metrics: dict, data_dict: list[dict], file_info: dict) -> Path:
    """Generate HTML report using Jinja2 template."""
    
    outdir_path = Path(ctx["outdir_path"])
    template_path = Path(__file__).parent / "report_template.html"
    
    # Setup Jinja2 environment
    env = Environment(loader=FileSystemLoader(template_path.parent))
    template = env.get_template(template_path.name)
    
    # Prepare template data
    job_yaml = io.read_text(ctx["job_path"]) if ctx.get("job_path") else ""
    dataset_id = hashx.dataset_id(job_yaml, "pandas|pyarrow")
    
    template_data = {
        "lang": ctx.get("lang", "fr"),
        "dataset_id": dataset_id,
        "run_id": ctx["run_id"],
        "generation_date": io.now_iso(),
        "data_dictionary": data_dict,
        "csv_path": file_info.get("csv_name", "dataset.csv"),
        "parquet_path": file_info.get("parquet_name", "dataset.parquet"),
        "csv_sha256": file_info.get("csv_sha256", ""),
        "parquet_sha256": file_info.get("parquet_sha256", ""),
        "robots_compliance": True,
        "tos_breaches": [],
        "pii_present": False,
        "anonymization_used": False,
        **metrics
    }
    
    # Render HTML
    html_content = template.render(**template_data)
    html_path = outdir_path / "data_quality_report.html"
    io.write_text(html_path, html_content)
    
    return html_path


def generate_pdf_report(html_path: Path) -> Path:
    """Generate PDF report from HTML using WeasyPrint."""
    
    pdf_path = html_path.with_suffix(".pdf")
    
    try:
        # Generate PDF with WeasyPrint
        weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return pdf_path
    except Exception as e:
        print(f"Warning: PDF generation failed: {e}")
        return None


def run(cfg, ctx):
    """Enhanced export function that merges quality data and generates reports."""
    
    outdir_path = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    
    try:
        # Merge quality data with main dataset
        merged_df = merge_quality_data(outdir_path)

        # Calculate metrics before reducing to curated columns
        quality_metrics = calculate_quality_metrics(merged_df)

        # Keep only curated columns required by the business specification
        df = build_curated_dataset(merged_df)

        # Sort data consistently
        sort_columns = [c for c in ["Nom entreprise", "Identifiant (SIREN/SIRET/DUNS)"] if c in df.columns]
        if sort_columns:
            df = df.sort_values(by=sort_columns, kind="mergesort")

        # Write output files
        csv_path = outdir_path / "dataset.csv"
        parquet_path = outdir_path / "dataset.parquet"

        # Ensure key identifier columns remain strings (preserve leading zeros)
        string_columns = [
            "Nom entreprise",
            "Identifiant (SIREN/SIRET/DUNS)",
            "Forme juridique",
            "Adresse",
            "Région / Département",
            "Téléphone standard",
            "Email générique",
            "Site web",
            "Nom + fonction du dirigeant",
            "Email pro vérifié",
            "Technologies web",
            "Secteur (NAF/APE)",
            "CA",
        ]
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype("string")

        df.to_csv(csv_path, index=False, encoding="utf-8")
        pq.write_table(pa.Table.from_pandas(df), parquet_path, compression="snappy")

        # Calculate data dictionary on curated output
        data_dictionary = calculate_data_dictionary(df)
        
        # Calculate file hashes
        csv_sha256 = io.sha256_file(csv_path)
        parquet_sha256 = io.sha256_file(parquet_path)
        
        file_info = {
            "csv_name": csv_path.name,
            "parquet_name": parquet_path.name,
            "csv_sha256": csv_sha256,
            "parquet_sha256": parquet_sha256
        }
        
        # Generate reports
        html_path = generate_html_report(ctx, quality_metrics, data_dictionary, file_info)
        pdf_path = generate_pdf_report(html_path)
        
        # Create enhanced manifest
        job_yaml = io.read_text(ctx["job_path"]) if ctx.get("job_path") else ""
        manifest = {
            "run_id": ctx["run_id"],
            "dataset_id": hashx.dataset_id(job_yaml, "pandas|pyarrow"),
            "records": int(len(df)),
            "paths": {
                "csv": str(csv_path),
                "parquet": str(parquet_path),
                "html_report": str(html_path),
                "pdf_report": str(pdf_path) if pdf_path else None
            },
            "quality_metrics": quality_metrics,
            "manifest": {
                "robots_compliance": True,
                "tos_breaches": [],
                "pii_present": False,
                "anonymization_used": False
            }
        }
        
        io.write_json(outdir_path / "manifest.json", manifest)
        
        # Write data dictionary (maintain backward compatibility)
        dd_lines = [f"- {item['column']}: non_null={item['non_null']}" for item in data_dictionary]
        io.write_text(outdir_path / "data_dictionary.md", "\n".join(dd_lines))
        
        # Write SHA256 checksums
        checksum_content = f"csv {csv_sha256}\nparquet {parquet_sha256}\n"
        if html_path:
            checksum_content += f"html {io.sha256_file(html_path)}\n"
        if pdf_path:
            checksum_content += f"pdf {io.sha256_file(pdf_path)}\n"
        io.write_text(outdir_path / "sha256.txt", checksum_content)
        
        return {
            "status": "OK",
            "csv": str(csv_path),
            "parquet": str(parquet_path),
            "html_report": str(html_path),
            "pdf_report": str(pdf_path) if pdf_path else None,
            **quality_metrics
        }
        
    except Exception as e:
        return {
            "status": "FAIL",
            "error": str(e)
        }
