"""
NAF Reference Collection Module

This module downloads and processes NAF (Nomenclature d'Activités Françaises) reference data
from INSEE APIs or SIRENE open data sources, filtered by the requested NAF code.

NAF is the French classification of economic activities.
"""

from pathlib import Path
import logging
import pandas as pd
from typing import Optional, Dict, Any

from utils import io
from utils.http import get_json, stream_download, HttpError
from utils.parquet import ParquetBatchWriter

LOGGER = logging.getLogger("collect.nafreference")

# INSEE API endpoints for NAF reference data
INSEE_NAF_API_BASE = "https://api.insee.fr/metadonnees/V1/codes/nafr2"
SIRENE_OPEN_DATA_BASE = "https://files.data.gouv.fr/insee-sirene"

# Fallback NAF data URLs (open data)
NAF_REFERENCE_URLS = {
    "naf_2008": "https://www.insee.fr/fr/statistiques/fichier/2120875/naf2008_liste_n5.xls",
    "naf_sections": "https://www.insee.fr/fr/statistiques/fichier/2120875/naf2008_sections.xls"
}


def _filter_naf_data(df: pd.DataFrame, naf_filter: Optional[str] = None) -> pd.DataFrame:
    """Filter NAF dataframe by NAF code prefix if provided."""
    if not naf_filter:
        return df
    
    # Look for NAF code columns
    naf_columns = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['naf', 'code', 'classe', 'sous-classe']):
            naf_columns.append(col)
    
    if not naf_columns:
        LOGGER.warning("No NAF code column found in data, returning unfiltered data")
        return df
    
    naf_col = naf_columns[0]  # Use first matching column
    
    # Convert NAF filter and column to string and clean them
    naf_filter_clean = str(naf_filter).strip()
    
    # Filter the dataframe
    mask = df[naf_col].astype(str).str.replace('.', '').str.replace(' ', '').str.startswith(naf_filter_clean.replace('.', ''))
    filtered_df = df[mask]
    
    LOGGER.info(f"Filtered NAF data from {len(df)} to {len(filtered_df)} rows with filter '{naf_filter}'")
    return filtered_df


def _download_insee_naf_data(naf_code: Optional[str] = None, timeout: float = 30.0) -> Optional[pd.DataFrame]:
    """Download NAF reference data from INSEE API."""
    try:
        # Try to get data from INSEE API
        url = f"{INSEE_NAF_API_BASE}/codes"
        if naf_code:
            url = f"{INSEE_NAF_API_BASE}/codes/{naf_code}"
        
        data = get_json(url, timeout=timeout)
        
        # Convert to DataFrame
        if isinstance(data, dict) and 'codes' in data:
            df = pd.DataFrame(data['codes'])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
            
        return df
        
    except HttpError as e:
        LOGGER.warning(f"Failed to fetch from INSEE API: {e}")
        return None


def _download_fallback_naf_data(outdir: Path, naf_filter: Optional[str] = None, timeout: float = 60.0) -> Optional[Path]:
    """Download NAF reference data from fallback open data sources."""
    try:
        # Download the main NAF reference file
        naf_url = NAF_REFERENCE_URLS["naf_2008"]
        temp_file = outdir / "naf_reference_raw.xls"
        
        LOGGER.info(f"Downloading NAF reference data from {naf_url}")
        downloaded_file = stream_download(naf_url, temp_file, timeout=timeout)
        
        # Read the Excel file
        df = pd.read_excel(downloaded_file, sheet_name=0)
        
        # Clean up column names and data
        df = df.dropna(how='all')  # Remove empty rows
        
        # The INSEE Excel file typically has NAF code in first column, description in second
        if len(df.columns) >= 2:
            # Rename columns to standard names
            new_columns = ['code_naf', 'libelle']
            if len(df.columns) > 2:
                new_columns.extend([f'col_{i}' for i in range(2, len(df.columns))])
            df.columns = new_columns[:len(df.columns)]
            
            # Clean the data
            df['code_naf'] = df['code_naf'].astype(str).str.strip()
            df['libelle'] = df['libelle'].astype(str).str.strip()
            
            # Remove header rows that might be included
            df = df[df['code_naf'].str.match(r'^[0-9]', na=False)]
        
        # Filter if NAF code is specified
        df = _filter_naf_data(df, naf_filter)
        
        # Save as parquet
        output_path = outdir / "naf_reference.parquet"
        with ParquetBatchWriter(output_path) as writer:
            writer.write_pandas(df, preserve_index=False)
        
        # Clean up temporary file
        temp_file.unlink(missing_ok=True)
        
        return output_path
        
    except Exception as e:
        LOGGER.error(f"Failed to download fallback NAF data: {e}")
        return None


def run(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for NAF reference collection.
    
    Configuration options:
    - nafreference.naf_code: Specific NAF code to filter by (optional)
    - nafreference.timeout: Request timeout in seconds (default: 60)
    - nafreference.source: Data source preference ('insee' or 'fallback', default: 'auto')
    
    Returns:
    - file: Path to the generated parquet file
    - status: Processing status
    - records_count: Number of records in the output
    """
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir")) / "nafreference"
    io.ensure_dir(outdir)
    
    # Get configuration
    naf_config = cfg.get("nafreference") or {}
    naf_code = naf_config.get("naf_code")
    timeout = float(naf_config.get("timeout", 60))
    source_preference = naf_config.get("source", "auto")
    
    if ctx.get("dry_run"):
        path = outdir / "empty.parquet"
        # Create empty parquet with minimal schema
        empty_df = pd.DataFrame({"code_naf": [], "libelle": []})
        with ParquetBatchWriter(path) as writer:
            writer.write_pandas(empty_df, preserve_index=False)
        return {"file": str(path), "status": "DRY_RUN"}
    
    output_file = None
    df = None
    
    # Try INSEE API first if preferred or auto
    if source_preference in ("insee", "auto"):
        LOGGER.info("Attempting to fetch NAF data from INSEE API")
        df = _download_insee_naf_data(naf_code, timeout)
        
        if df is not None and not df.empty:
            output_file = outdir / "naf_reference_insee.parquet"
            with ParquetBatchWriter(output_file) as writer:
                writer.write_pandas(df, preserve_index=False)
    
    # Fall back to open data sources if INSEE API failed or fallback preferred
    if (output_file is None or source_preference == "fallback"):
        LOGGER.info("Using fallback open data sources for NAF reference")
        output_file = _download_fallback_naf_data(outdir, naf_code, timeout)
        
        if output_file and output_file.exists():
            # Read back to get record count
            df = pd.read_parquet(output_file)
    
    if output_file is None or not output_file.exists():
        return {"status": "FAILED", "reason": "NO_DATA_SOURCE_AVAILABLE"}
    
    records_count = len(df) if df is not None else 0
    
    return {
        "file": str(output_file),
        "status": "OK",
        "records_count": records_count,
        "naf_filter": naf_code if naf_code else "all"
    }