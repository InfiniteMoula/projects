
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa

from utils.parquet import ParquetBatchWriter, iter_batches

E164_FR = re.compile(r"^\+33\d{9}$")

# Placeholder values that should be treated as missing/invalid data
PLACEHOLDER_VALUES = {
    "TELEPHONE NON RENSEIGNE",
    "ADRESSE NON RENSEIGNEE", 
    "DENOMINATION NON RENSEIGNEE"
}


def _as_str(s: pd.Series) -> pd.Series:
    return s.astype("string")


def run(cfg, ctx):
    t0 = time.time()
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    
    # Primary input: Google Maps enriched data
    maps_inp = outdir / "google_maps_enriched.parquet"
    # Fallback candidates
    candidates = [
        outdir / "enriched_email.parquet",
        outdir / "enriched_dns.parquet",
        outdir / "enriched_domain.parquet",
        outdir / "normalized.parquet",
    ]
    
    src = maps_inp if maps_inp.exists() else next((p for p in candidates if p.exists()), None)
    if not src:
        return {"status": "WARN", "error": "no input for phone checks"}

    outp = outdir / "enriched_phone.parquet"
    total = 0
    logger = ctx.get("logger")
    using_maps_data = src == maps_inp

    if logger:
        logger.info(f"Phone checks using {'Google Maps' if using_maps_data else 'fallback'} data")

    try:
        with ParquetBatchWriter(outp) as writer:
            for pdf in iter_batches(src):
                if pdf.empty:
                    continue
                    
                if "telephone_norm" not in pdf.columns:
                    pdf["telephone_norm"] = pd.NA
                tel = _as_str(pdf["telephone_norm"]).fillna("")
                
                # If using Google Maps data, prioritize maps phone numbers
                if using_maps_data and "maps_phone_numbers" in pdf.columns:
                    maps_phones = _as_str(pdf["maps_phone_numbers"]).fillna("")
                    
                    # Parse multiple phone numbers from maps (separated by '; ')
                    for idx, phones_str in maps_phones.items():
                        if phones_str and isinstance(phones_str, str):
                            phones = [p.strip() for p in phones_str.split(';') if p.strip()]
                            for phone in phones:
                                # Clean and normalize phone number
                                clean_phone = re.sub(r'[^\d+]', '', phone)
                                # Convert French national format to international
                                if clean_phone.startswith('0') and len(clean_phone) == 10:
                                    clean_phone = '+33' + clean_phone[1:]
                                elif clean_phone.startswith('33') and len(clean_phone) == 11:
                                    clean_phone = '+' + clean_phone
                                
                                # Use first valid phone number found
                                if E164_FR.match(clean_phone) and tel.loc[idx] == "":
                                    tel.loc[idx] = clean_phone
                                    pdf.loc[idx, "telephone_norm"] = clean_phone
                                    break
                    
                    if logger:
                        maps_phones_found = (tel != "").sum()
                        logger.debug(f"Found {maps_phones_found} phones from Google Maps data")
                
                # Check if phone is valid E164 format AND not a placeholder value
                is_e164_format = tel.str.match(E164_FR)
                is_placeholder = tel.isin(PLACEHOLDER_VALUES)
                pdf["phone_valid"] = (is_e164_format & ~is_placeholder).astype("boolean")
                
                table = pa.Table.from_pandas(pdf, preserve_index=False)
                writer.write_table(table)
                total += len(pdf)

        return {
            "status": "OK", 
            "file": str(outp), 
            "rows": total, 
            "duration_s": round(time.time() - t0, 3),
            "using_maps_data": using_maps_data
        }
    except Exception as exc:
        if logger:
            logger.exception("phone checks failed: %s", exc)
        return {"status": "FAIL", "error": str(exc), "duration_s": round(time.time() - t0, 3)}
        if logger:
            logger.exception("phone checks failed: %s", exc)
        return {"status": "FAIL", "error": str(exc), "duration_s": round(time.time() - t0, 3)}
