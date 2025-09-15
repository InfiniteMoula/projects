
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa

from utils.parquet import ParquetBatchWriter, iter_batches

E164_FR = re.compile(r"^\+33\d{9}$")


def _as_str(s: pd.Series) -> pd.Series:
    return s.astype("string")


def run(cfg, ctx):
    t0 = time.time()
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    candidates = [
        outdir / "enriched_email.parquet",
        outdir / "enriched_dns.parquet",
        outdir / "enriched_domain.parquet",
        outdir / "normalized.parquet",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if not src:
        return {"status": "WARN", "error": "no input for phone checks"}

    outp = outdir / "enriched_phone.parquet"
    total = 0
    logger = ctx.get("logger")

    try:
        with ParquetBatchWriter(outp) as writer:
            for pdf in iter_batches(src):
                if pdf.empty:
                    continue
                if "telephone_norm" not in pdf.columns:
                    pdf["telephone_norm"] = pd.NA
                tel = _as_str(pdf["telephone_norm"]).fillna("")
                pdf["phone_valid"] = tel.str.match(E164_FR).astype("boolean")
                table = pa.Table.from_pandas(pdf, preserve_index=False)
                writer.write_table(table)
                total += len(pdf)

        return {"status": "OK", "file": str(outp), "rows": total, "duration_s": round(time.time() - t0, 3)}
    except Exception as exc:
        if logger:
            logger.exception("phone checks failed: %s", exc)
        return {"status": "FAIL", "error": str(exc), "duration_s": round(time.time() - t0, 3)}
