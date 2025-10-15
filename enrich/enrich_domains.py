from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from net.http_client import RequestLimiter

from features import embeddings
from utils.scoring import score_domain
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


def _strip_html(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_candidate_text(candidate: Mapping[str, object]) -> str:
    for key in ("text", "homepage", "content", "body", "html", "snippet", "description"):
        value = candidate.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if key == "html":
            text = _strip_html(text)
        return text
    return ""


def _iter_semantic_candidates(raw: object) -> List[Dict[str, str]]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        items = [raw]
    elif isinstance(raw, (str, bytes)):
        return []
    elif isinstance(raw, Iterable):
        items = list(raw)
    else:
        return []

    candidates: List[Dict[str, str]] = []
    for item in items:
        if isinstance(item, Mapping):
            url = str(
                item.get("url")
                or item.get("site")
                or item.get("homepage")
                or item.get("domain")
                or ""
            ).strip()
            if not url:
                continue
            text = _extract_candidate_text(item)
            if not text:
                continue
            source = str(item.get("source") or item.get("origin") or "semantic").strip() or "semantic"
            title = str(item.get("title") or item.get("page_title") or "").strip()
            candidates.append({
                "url": url,
                "text": text,
                "source": source,
                "title": title,
            })
    return candidates


def _apply_semantic_selection(
    df: pd.DataFrame,
    embeddings_cfg: Mapping[str, object],
    *,
    logger: Optional[Any] = None,
) -> Tuple[pd.DataFrame, List[Any]]:
    candidate_column = None
    for column in ("site_web_candidates", "domain_candidates", "website_candidates"):
        if column in df.columns:
            candidate_column = column
            break
    if not candidate_column:
        return df, []

    try:
        enabled = bool(embeddings_cfg.get("enabled", False))
    except Exception:
        enabled = False
    if not enabled:
        return df, []

    threshold_raw = embeddings_cfg.get("threshold", 0.6)
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError):
        threshold = 0.6
    threshold = max(0.0, min(threshold, 1.0))

    model_name = str(embeddings_cfg.get("model") or embeddings.DEFAULT_MODEL)

    work = df.copy()
    for column in ("site_web", "site_web_source", "site_web_score"):
        if column not in work.columns:
            work[column] = pd.NA
    semantic_col = "site_web_semantic_similarity"
    if semantic_col not in work.columns:
        work[semantic_col] = pd.NA

    selected_indices: List[Any] = []

    for idx, series in work.iterrows():
        raw_candidates = series.get(candidate_column)
        candidates = _iter_semantic_candidates(raw_candidates)
        if len(candidates) < 2:
            continue

        company_text = embeddings.generate_text(series)
        if not company_text:
            continue
        company_vec = embeddings.embed(company_text, model_name=model_name)
        if company_vec.size == 0 or float(np.linalg.norm(company_vec)) == 0.0:
            continue

        best_score = 0.0
        best_candidate: Optional[Dict[str, str]] = None
        for candidate in candidates:
            candidate_vec = embeddings.embed(candidate["text"], model_name=model_name)
            similarity = embeddings.cosine(company_vec, candidate_vec)
            if similarity > best_score:
                best_score = similarity
                best_candidate = candidate

        if not best_candidate or best_score < threshold:
            continue

        url = best_candidate["url"]
        source = best_candidate["source"]
        source_label = source if source.upper().startswith("SEMANTIC") else f"SEMANTIC:{source}"
        title = best_candidate.get("title", "")

        denomination = str(
            series.get("denomination")
            or series.get("raison_sociale")
            or series.get("enseigne")
            or series.get("nom")
            or ""
        )
        city = str(series.get("ville") or series.get("commune") or series.get("city") or "")

        site_score = score_domain(url, denomination or "", city or "", title)

        work.at[idx, "site_web"] = url
        work.at[idx, "site_web_source"] = source_label
        work.at[idx, "site_web_score"] = site_score
        work.at[idx, semantic_col] = float(best_score)
        selected_indices.append(idx)

        if logger:
            logger.debug(
                "semantic domain match idx=%s url=%s similarity=%.3f threshold=%.3f",
                idx,
                url,
                best_score,
                threshold,
            )

    return work, selected_indices


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
    shared_cfg.pop("embeddings", None)

    embeddings_cfg = enrichment_cfg.get("embeddings") or {}
    df_after_semantic, semantic_indices = _apply_semantic_selection(df_in, embeddings_cfg, logger=logger)
    if logger and semantic_indices:
        logger.info("Semantic domain selection accepted %s candidates", len(semantic_indices))

    df_pending = df_after_semantic.drop(index=semantic_indices, errors="ignore") if semantic_indices else df_after_semantic
    pending_rows = len(df_pending)

    if pending_rows == 0:
        df_out = df_after_semantic
    elif pending_rows <= chunk_size or worker_limit <= 1:
        pending_result = site_web_search.run(df_pending, dict(shared_cfg))
        df_out = df_after_semantic.copy()
        for column in pending_result.columns:
            df_out.loc[pending_result.index, column] = pending_result[column]
    else:
        chunk_results: Dict[int, pd.DataFrame] = {}
        worker_count = min(worker_limit, max(1, (pending_rows + chunk_size - 1) // chunk_size))

        def _process_chunk(chunk_index: int, chunk_df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
            chunk_cfg = dict(shared_cfg)
            result = site_web_search.run(chunk_df, chunk_cfg)
            return chunk_index, result

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_process_chunk, chunk_idx, chunk_df): chunk_idx
                for chunk_idx, chunk_df in _iter_chunks(df_pending, chunk_size)
            }
            for future in as_completed(futures):
                chunk_idx, chunk_df = future.result()
                chunk_results[chunk_idx] = chunk_df
                del chunk_df

        ordered_chunks = [chunk_results[idx] for idx in sorted(chunk_results)]
        combined = pd.concat(ordered_chunks)
        combined = combined.reindex(df_pending.index)
        df_out = df_after_semantic.copy()
        for column in combined.columns:
            df_out.loc[combined.index, column] = combined[column]

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
