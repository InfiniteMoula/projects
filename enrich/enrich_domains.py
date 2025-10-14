from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Mapping, Tuple

import pandas as pd

from . import site_web_search


def _load_input(outdir: Path) -> Tuple[pd.DataFrame, Path]:
    candidates = [
        outdir / "normalized.parquet",
        outdir / "normalized.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.suffix == ".parquet":
                return pd.read_parquet(candidate), candidate
            return pd.read_csv(candidate), candidate
    raise FileNotFoundError("normalized dataset not found")


def run(cfg: Dict, ctx: Dict) -> Dict[str, object]:
    logger = ctx.get("logger")
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    start = time.time()

    try:
        df_in, source_path = _load_input(outdir)
    except FileNotFoundError as exc:
        if logger:
            logger.warning("enrich.domains skipped: %s", exc)
        return {"status": "SKIPPED", "reason": "NO_NORMALIZED_DATA"}

    if df_in.empty:
        if logger:
            logger.info("enrich.domains skipped: empty dataset (%s)", source_path.name)
        return {"status": "SKIPPED", "reason": "EMPTY_INPUT"}

    enrichment_cfg: Mapping[str, object] = (ctx.get("enrichment_config") or {}).get("domains") or {}
    if logger:
        logger.info("Running domain enrichment on %s rows (source=%s)", len(df_in), source_path.name)

    df_out = site_web_search.run(df_in, enrichment_cfg)

    output_path = outdir / "domains_enriched.parquet"
    csv_path = outdir / "domains_enriched.csv"
    df_out.to_parquet(output_path, index=False)
    df_out.to_csv(csv_path, index=False)

    sites_found = int(df_out.get("site_web", pd.Series(dtype="string")).fillna("").astype("string").str.strip().ne("").sum())
    duration = round(time.time() - start, 3)

    if logger:
        logger.info("enrich.domains completed: %s/%s websites identified in %ss", sites_found, len(df_out), duration)

    return {
        "status": "OK",
        "file": str(output_path),
        "rows": len(df_out),
        "sites_found": sites_found,
        "source": str(source_path),
        "duration_s": duration,
    }


__all__ = ["run"]
