
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def _analyze_duplicates(df, keys, logger=None):
    """Analyze duplicate patterns to validate deduplication criteria."""
    analysis = {}
    
    # Check data completeness for each key
    key_completeness = {}
    for key in keys:
        if key in df.columns:
            non_empty = (df[key].notna() & (df[key] != "")).sum()
            key_completeness[key] = {
                "total_records": len(df),
                "non_empty_records": int(non_empty),
                "completeness_pct": round((non_empty / len(df)) * 100, 2)
            }
    
    # Analyze duplicate patterns by key combination
    duplicate_patterns = {}
    for i in range(1, len(keys) + 1):
        from itertools import combinations
        for key_combo in combinations(keys, i):
            existing_combo = [k for k in key_combo if k in df.columns]
            if existing_combo:
                # Create a mask for rows that have at least one non-empty value in this combo
                mask = df[existing_combo].apply(
                    lambda row: any(pd.notna(val) and val != "" for val in row), axis=1
                )
                subset_df = df[mask]
                
                if len(subset_df) > 0:
                    # Fill empty values for deduplication
                    subset_filled = subset_df.copy()
                    for col in existing_combo:
                        subset_filled[col] = subset_filled[col].fillna("").astype(str)
                    
                    before_dedupe = len(subset_filled)
                    after_dedupe = len(subset_filled.drop_duplicates(subset=existing_combo))
                    duplicates = before_dedupe - after_dedupe
                    
                    duplicate_patterns["|".join(existing_combo)] = {
                        "applicable_records": before_dedupe,
                        "unique_records": after_dedupe,
                        "duplicates_found": duplicates,
                        "duplicate_rate_pct": round((duplicates / before_dedupe) * 100, 2) if before_dedupe > 0 else 0
                    }
    
    # Find most common duplicate combinations
    df_filled = df.copy()
    for col in keys:
        if col in df_filled.columns:
            df_filled[col] = df_filled[col].fillna("").astype(str)
    
    existing_keys = [k for k in keys if k in df.columns]
    if existing_keys:
        duplicate_mask = df_filled.duplicated(subset=existing_keys, keep=False)
        duplicates_df = df_filled[duplicate_mask]
        
        if len(duplicates_df) > 0:
            # Group by duplicate key combinations to see common patterns
            common_patterns = duplicates_df.groupby(existing_keys).size().sort_values(ascending=False).head(10)
            analysis["top_duplicate_patterns"] = {}
            for pattern, count in common_patterns.items():
                pattern_key = "|".join([f"{k}:{v}" for k, v in zip(existing_keys, pattern)])
                analysis["top_duplicate_patterns"][pattern_key] = int(count)
    
    analysis["key_completeness"] = key_completeness
    analysis["duplicate_patterns"] = duplicate_patterns
    
    # Log warnings for suspicious patterns
    if logger:
        total_records = len(df)
        for combo, stats in duplicate_patterns.items():
            dup_rate = stats["duplicate_rate_pct"]
            if dup_rate > 50:
                logger.warning(f"High duplicate rate for keys {combo}: {dup_rate}% ({stats['duplicates_found']} of {stats['applicable_records']} records)")
            elif dup_rate > 20:
                logger.info(f"Moderate duplicate rate for keys {combo}: {dup_rate}% ({stats['duplicates_found']} of {stats['applicable_records']} records)")
        
        # Check for suspicious empty value patterns
        for key, stats in key_completeness.items():
            if stats["completeness_pct"] < 30:
                logger.warning(f"Low data completeness for key '{key}': {stats['completeness_pct']}% ({stats['non_empty_records']} of {stats['total_records']} records)")
    
    return analysis


def run(cfg, ctx):
    keys = (cfg.get("dedupe") or {}).get("keys", ["siren", "domain_root", "best_email", "telephone_norm"])
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    source_candidates = [outdir / "enriched_email.parquet", outdir / "normalized.parquet"]
    src = next((p for p in source_candidates if p.exists()), None)
    if not src:
        return {"status": "WARN", "error": "no input for dedupe"}

    table = pq.read_table(src)
    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    existing_keys = [k for k in keys if k in df.columns]
    if not existing_keys:
        return {"status": "WARN", "error": "no dedupe keys present"}

    logger = ctx.get("logger")
    before = len(df)
    
    # Analyze duplicate patterns before deduplication
    if logger:
        logger.info(f"Starting deduplication analysis for {before} records using keys: {existing_keys}")
    
    duplicate_analysis = _analyze_duplicates(df, existing_keys, logger)
    
    # Improved deduplication logic
    # Instead of just fillna(""), we handle empty values more intelligently
    df_for_dedupe = df.copy()
    
    # Fill NaN values with empty strings for deduplication, but preserve the information
    for col in existing_keys:
        df_for_dedupe[col] = df_for_dedupe[col].fillna("").astype(str)
    
    # Only consider records that have at least one non-empty key value for deduplication
    # This prevents false positives from records that are completely empty
    has_meaningful_data = df_for_dedupe[existing_keys].apply(
        lambda row: any(val != "" for val in row), axis=1
    )
    
    if has_meaningful_data.sum() == 0:
        if logger:
            logger.warning("No records have meaningful data in deduplication keys")
        return {
            "status": "WARN", 
            "error": "no records with meaningful deduplication key data",
            "before": before,
            "after": before,
            "duplicate_analysis": duplicate_analysis
        }
    
    # Split into records with meaningful data and completely empty records
    meaningful_df = df_for_dedupe[has_meaningful_data]
    empty_df = df_for_dedupe[~has_meaningful_data]
    
    # Deduplicate only the meaningful records
    meaningful_deduped = meaningful_df.drop_duplicates(subset=existing_keys)
    
    # Combine back with empty records (keeping all empty records as they're not really duplicates)
    final_df = pd.concat([meaningful_deduped, empty_df], ignore_index=True)
    
    after = len(final_df)
    duplicates_removed = before - after
    duplicate_rate = (duplicates_removed / before * 100) if before > 0 else 0
    
    # Log results
    if logger:
        logger.info(f"Deduplication completed: {before} → {after} records ({duplicates_removed} duplicates removed, {duplicate_rate:.2f}%)")
        if duplicate_rate > 30:
            logger.warning(f"High duplicate rate detected ({duplicate_rate:.2f}%), review deduplication criteria")
        elif duplicate_rate > 10:
            logger.info(f"Moderate duplicate rate ({duplicate_rate:.2f}%), within normal range")
    
    out_path = outdir / "deduped.parquet"
    pq.write_table(pa.Table.from_pandas(final_df, preserve_index=False), out_path, compression="snappy")
    
    return {
        "status": "OK", 
        "file": str(out_path), 
        "before": before, 
        "after": after,
        "duplicates_removed": duplicates_removed,
        "duplicate_rate_pct": round(duplicate_rate, 2),
        "total_records": before,  # For KPI tracking compatibility
        "duplicate_analysis": duplicate_analysis
    }
