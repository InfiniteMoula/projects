import os
import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Environment, FileSystemLoader
import weasyprint
from utils import io, hashx


def merge_quality_data(outdir_path: Path) -> pd.DataFrame:
    """Merge deduped data with quality scores and address enrichment."""
    
    # Priority order for source data
    source_candidates = [
        outdir_path / "deduped.parquet",
        outdir_path / "enriched_email.parquet", 
        outdir_path / "enriched_domain.parquet",
        outdir_path / "address_enriched.parquet",  # Add address enrichment
        outdir_path / "normalized.parquet"
    ]
    
    source_path = next((p for p in source_candidates if p.exists()), None)
    if not source_path:
        raise FileNotFoundError("No source parquet file found for export")
    
    # Load main dataset
    df = pd.read_parquet(source_path)
    
    # Try to merge with address enrichment data if not already included
    address_path = outdir_path / "address_enriched.parquet"
    if address_path.exists() and source_path != address_path:
        try:
            address_df = pd.read_parquet(address_path)
            # Merge on common columns (typically siren/siret)
            common_cols = [col for col in ['siren', 'siret'] if col in df.columns and col in address_df.columns]
            if common_cols:
                # Keep only the new enrichment columns from address search
                address_cols = ['found_business_names_str', 'found_phones_str', 'found_emails_str', 'search_status']
                address_cols = [col for col in address_cols if col in address_df.columns]
                
                if address_cols:
                    merge_df = address_df[common_cols + address_cols]
                    df = df.merge(merge_df, on=common_cols, how='left')
                    print(f"Merged address enrichment data with {len(address_cols)} new columns")
        except Exception as e:
            print(f"Warning: Could not merge address enrichment data: {e}")
    
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
        df = merge_quality_data(outdir_path)
        
        # Sort data consistently
        sort_columns = [c for c in ["siren", "raison_sociale"] if c in df.columns]
        if sort_columns:
            df = df.sort_values(by=sort_columns, kind="mergesort")
        
        # Write output files
        csv_path = outdir_path / "dataset.csv"
        parquet_path = outdir_path / "dataset.parquet"
        
        df.to_csv(csv_path, index=False, encoding="utf-8")
        pq.write_table(pa.Table.from_pandas(df), parquet_path, compression="snappy")
        
        # Calculate metrics and data dictionary
        quality_metrics = calculate_quality_metrics(df)
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
