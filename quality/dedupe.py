
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def run(cfg, ctx):
    keys = (cfg.get("dedupe") or {}).get("keys", ["siren", "domain_root", "best_email", "telephone_norm"])
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    source_candidates = [outdir / "enriched_email.parquet", outdir / "normalized.parquet"]
    src = next((p for p in source_candidates if p.exists()), None)
    if not src:
        return {"status": "WARN", "error": "no input for dedupe"}

    table = pq.read_table(src)
    df = table.to_pandas(types_mapper=pd.ArrowDtype).fillna("")
    existing_keys = [k for k in keys if k in df.columns]
    if not existing_keys:
        return {"status": "WARN", "error": "no dedupe keys present"}

    before = len(df)
    df = df.drop_duplicates(subset=existing_keys)
    out_path = outdir / "deduped.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path, compression="snappy")
    return {"status": "OK", "file": str(out_path), "before": before, "after": len(df)}
