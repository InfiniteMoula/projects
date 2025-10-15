from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd

from net.http_client import RequestLimiter

from . import site_web_search

DEFAULT_CHUNK_SIZE = 300
DEFAULT_MAX_WORKERS = 8


def _resolve_chunk_size(cfg: Optional[Mapping[str, object]], enrichment_cfg: Mapping[str, object]) -> int:
    for source in (cfg or {}, enrichment_cfg):
        value = source.get("chunk_size") if isinstance(source, Mapping) else None
        if value is None:
            continue
        try:
            size = int(value)
        except (TypeError, ValueError):
            continue
        if size <= 0:
            continue
        return max(200, min(size, 500))
    return DEFAULT_CHUNK_SIZE


def _resolve_max_workers(cfg: Optional[Mapping[str, object]], enrichment_cfg: Mapping[str, object]) -> int:
    for source in (cfg or {}, enrichment_cfg):
        value = source.get("max_workers") if isinstance(source, Mapping) else None
        if value is None:
            continue
        try:
            workers = int(value)
        except (TypeError, ValueError):
            continue
        if workers > 0:
            return workers
    return DEFAULT_MAX_WORKERS


def _iter_chunks(df: pd.DataFrame, chunk_size: int) -> Iterable[Tuple[int, pd.DataFrame]]:
    total = len(df)
    if chunk_size <= 0 or chunk_size >= total:
        yield 0, df
        return
    for chunk_index, start in enumerate(range(0, total, chunk_size)):
        stop = min(start + chunk_size, total)
        yield chunk_index, df.iloc[start:stop].copy()


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

    chunk_size = _resolve_chunk_size(cfg, enrichment_cfg)
    total_rows = len(df_in)
    worker_limit = _resolve_max_workers(cfg, enrichment_cfg)
    http_cfg = dict(enrichment_cfg.get("http_client", {}))
    max_concurrent_requests = max(1, int(http_cfg.get("max_concurrent_requests", worker_limit)))
    request_limiter = RequestLimiter(max_concurrent_requests)
    http_cfg["shared_request_limiter"] = request_limiter
    shared_cfg = dict(enrichment_cfg)
    shared_cfg["http_client"] = http_cfg

    if total_rows <= chunk_size or worker_limit <= 1:
        df_out = site_web_search.run(df_in, dict(shared_cfg))
    else:
        chunk_results: Dict[int, pd.DataFrame] = {}
        worker_count = min(worker_limit, max(1, (total_rows + chunk_size - 1) // chunk_size))

        def _process_chunk(chunk_index: int, chunk_df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
            # Copy config to keep thread-local view while sharing limiter and HTTP options
            chunk_cfg = dict(shared_cfg)
            result = site_web_search.run(chunk_df, chunk_cfg)
            return chunk_index, result

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_process_chunk, chunk_idx, chunk_df): chunk_idx
                for chunk_idx, chunk_df in _iter_chunks(df_in, chunk_size)
            }
            for future in as_completed(futures):
                chunk_idx, chunk_df = future.result()
                chunk_results[chunk_idx] = chunk_df
                # Release memory early
                del chunk_df

        ordered_chunks = [chunk_results[idx] for idx in sorted(chunk_results)]
        df_out = pd.concat(ordered_chunks)
        df_out = df_out.reindex(df_in.index)

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
