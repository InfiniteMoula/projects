
from pathlib import Path

from utils import io
from utils.parquet import iter_batches


def run(cfg, ctx):
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    candidates = [outdir / "enriched_email.parquet", outdir / "normalized.parquet"]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        return {"status": "WARN", "error": "no data for quality checks"}

    issues = []
    siren_nulls = 0
    cp_bad = 0

    for df in iter_batches(source, columns=["siren", "cp"]):
        if "siren" in df.columns:
            siren_nulls += int(df["siren"].isna().sum())
        if "cp" in df.columns:
            cp_series = df["cp"].astype("string").fillna("")
            cp_bad += int((~cp_series.str.match(r"^\d{2,5}$", na=False)).sum())

    if siren_nulls:
        issues.append({"key": "siren_nulls", "count": siren_nulls})
    if cp_bad:
        issues.append({"key": "cp_format", "count": cp_bad})

    payload = {"issues": issues}
    out_path = outdir / "quality_checks.json"
    io.write_json(out_path, payload)
    return {"status": "OK", "file": str(out_path), "issues": issues}
