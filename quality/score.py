
# FILE: quality/score.py
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def run(cfg: dict, ctx: dict) -> dict:
    """
    Calcule un score agr?g? robuste (sans NaN) et ?crit :
      - quality_score.parquet (une colonne score_quality)
      - quality_summary.json  (m?triques agr?g?es)
    Source pr?f?r?e : deduped.parquet (sinon enriched_email.parquet, sinon enriched_domain.parquet)
    Les colonnes manquantes sont cr??es ? 0.
    """

    t0 = time.time()
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))

    candidates = [
        outdir / "deduped.parquet",
        outdir / "enriched_email.parquet",
        outdir / "enriched_domain.parquet",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if not src:
        return {"status": "WARN", "error": "no input parquet for scoring"}

    df = pq.read_table(src).to_pandas(types_mapper=pd.ArrowDtype)

    needed = ["contactability", "unicity", "completeness", "freshness"]
    for col in needed:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    weights = (cfg.get("scoring", {}) or {}).get("weights", {})
    w_contact = float(weights.get("contactability", 50))
    w_unicity = float(weights.get("unicity", 20))
    w_complete = float(weights.get("completeness", 20))
    w_fresh = float(weights.get("freshness", 10))
    w_sum = max(w_contact + w_unicity + w_complete + w_fresh, 1.0)

    df["score_quality"] = (
        df["contactability"] * w_contact +
        df["unicity"] * w_unicity +
        df["completeness"] * w_complete +
        df["freshness"] * w_fresh
    ) / w_sum
    df["score_quality"] = pd.to_numeric(df["score_quality"], errors="coerce").fillna(0.0)

    out_parquet = outdir / "quality_score.parquet"
    out_json = outdir / "quality_summary.json"

    pq.write_table(pa.Table.from_pandas(df[["score_quality"]], preserve_index=False), out_parquet, compression="snappy")

    summary = {
        "rows": int(len(df)),
        "score_mean": float(df["score_quality"].mean(skipna=True) or 0.0),
        "score_p50": float(df["score_quality"].quantile(0.50, interpolation="linear")),
        "score_p90": float(df["score_quality"].quantile(0.90, interpolation="linear")),
        "duration_s": round(time.time() - t0, 3),
    }
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return {
        "status": "OK",
        "files": [str(out_parquet), str(out_json)],
        **summary,
    }
