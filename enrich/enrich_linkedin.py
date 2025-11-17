from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from net.http_client import HttpClient
from serp.providers import Result, SerpProvider, BingProvider, DuckDuckGoProvider, BraveProvider
from serp.playwright_provider import PlaywrightBingProvider, PlaywrightGoogleProvider

LOGGER = logging.getLogger("enrich.enrich_linkedin")

DEFAULT_PROVIDERS: Tuple[str, ...] = ("bing", "duckduckgo", "brave")
LEGAL_FORMS = (
    "SARL",
    "SARLU",
    "SAS",
    "SASU",
    "SA",
    "SCI",
    "SC",
    "SNC",
    "SCS",
    "SCA",
    "SCEA",
    "GIE",
    "EURL",
    "EARL",
    "SELARL",
    "EI",
    "EIRL",
    "AUTO-ENTREPRISE",
    "AUTO ENTREPRISE",
    "ASSOCIATION",
    "ASS",
    "COOPERATIVE",
    "COOP",
)

PROVIDER_REGISTRY = {
    "bing": BingProvider,
    "duckduckgo": DuckDuckGoProvider,
    "brave": BraveProvider,
    "playwright_bing": PlaywrightBingProvider,
    "playwright_google": PlaywrightGoogleProvider,
}


def process_linkedin(df_in: pd.DataFrame, cfg: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Locate LinkedIn company pages via SERP providers without direct scraping.
    """

    df_out = df_in.copy() if df_in is not None else pd.DataFrame()
    if df_out.empty:
        for col in ("linkedin_url", "linkedin_source", "linkedin_score"):
            if col not in df_out.columns:
                df_out[col] = pd.NA
        return df_out, {"matches": 0}

    provider_names: Tuple[str, ...] = tuple(cfg.get("providers", DEFAULT_PROVIDERS)) or DEFAULT_PROVIDERS
    provider_settings = cfg.get("providers_config", {})
    http_cfg = cfg.get("http_client", {})

    http_client = HttpClient(http_cfg)
    providers: List[Tuple[str, SerpProvider]] = []
    for name in provider_names:
        key = str(name).lower()
        provider_cls = PROVIDER_REGISTRY.get(key)
        if not provider_cls:
            LOGGER.warning("Unknown SERP provider %s; skipping", name)
            continue
        settings = {}
        if isinstance(provider_settings, Mapping):
            settings = provider_settings.get(name, provider_settings.get(key, {})) or {}
        providers.append((key, provider_cls(settings, http_client)))

    if not providers:
        LOGGER.warning("No SERP providers configured; LinkedIn enrichment will be skipped")
        for col in ("linkedin_url", "linkedin_source", "linkedin_score"):
            if col not in df_out.columns:
                df_out[col] = pd.NA
        http_client.close()
        return df_out, {"matches": 0}

    for col in ("linkedin_url", "linkedin_source", "linkedin_score"):
        if col not in df_out.columns:
            df_out[col] = pd.NA

    try:
        for row in df_out.itertuples(index=True):
            idx = row.Index
            raw_name = _safe_str(getattr(row, "denomination", ""))
            if not raw_name:
                continue
            raw_city = _safe_str(getattr(row, "ville", ""))

            norm_name = _normalize_text(raw_name)
            if not norm_name:
                continue
            norm_city = _normalize_text(raw_city)

            query_name = _prepare_search_text(raw_name)
            query_parts = [query_name]
            if norm_city:
                query_parts.append(norm_city)
            quoted = " ".join(query_parts).strip()
            query = f'site:linkedin.com/company "{quoted}"'

            best_result: Optional[Tuple[str, str, float]] = None  # url, source, score

            for provider_name, provider in providers:
                try:
                    results = provider.search(query)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Provider %s failed for %s: %s", provider_name, raw_name, exc)
                    continue

                candidate = _select_linkedin_result(results, norm_name)
                if not candidate:
                    continue
                url, score = candidate
                if not best_result or score > best_result[2]:
                    best_result = (url, provider_name, score)
                if best_result and best_result[2] >= 1.0:
                    break  # cannot beat perfect score

            if best_result:
                url, source, score = best_result
                df_out.at[idx, "linkedin_url"] = url
                df_out.at[idx, "linkedin_source"] = source
                df_out.at[idx, "linkedin_score"] = score
                LOGGER.info(
                    "LinkedIn found name=%s source=%s score=%.2f url=%s",
                    raw_name,
                    source,
                    score,
                    url,
                )
            else:
                LOGGER.info("LinkedIn not found name=%s", raw_name)
    finally:
        http_client.close()
        try:
            asyncio.run(http_client.aclose())
        except RuntimeError:
            # likely already in an event loop; ignore
            pass
    matches = int(df_out.get("linkedin_url", pd.Series(dtype="string")).fillna("").astype("string").str.strip().ne("").sum())
    return df_out, {"matches": matches}


def _select_linkedin_result(results: List[Result], norm_name: str) -> Optional[Tuple[str, float]]:
    best: Optional[Tuple[str, float]] = None
    for result in results:
        url = (result.url or "").strip()
        if not url:
            continue
        if "/company/" not in url or "/in/" in url:
            continue
        score = _score_result(result.title or "", norm_name)
        if score <= 0:
            continue
        if not best or score > best[1]:
            best = (url, score)
    return best


def _score_result(title: str, norm_name: str) -> float:
    if not norm_name:
        return 0.0
    norm_title = _normalize_text(title)
    if not norm_title:
        return 0.0
    if norm_title == norm_name:
        return 1.0
    if norm_name in norm_title or norm_title in norm_name:
        return 0.8
    return 0.0


def _prepare_search_text(text: str) -> str:
    stripped = _strip_accents(text).upper()
    stripped = _remove_legal_forms(stripped)
    stripped = re.sub(r"[^A-Z0-9\s-]", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.strip().lower()


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _remove_legal_forms(text: str) -> str:
    if not text:
        return ""
    pattern = r"\b(" + "|".join(sorted(LEGAL_FORMS, key=len, reverse=True)) + r")\b"
    return re.sub(pattern, " ", text, flags=re.IGNORECASE)


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    stripped = _strip_accents(text).lower()
    stripped = re.sub(r"[^a-z0-9]+", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.strip()


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def run(cfg: Dict, ctx: Dict) -> Dict[str, object]:
    logger = ctx.get("logger") or LOGGER
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    start = time.time()

    input_candidates = [
        outdir / "domains_enriched.parquet",
        outdir / "domains_enriched.csv",
        outdir / "normalized.parquet",
        outdir / "normalized.csv",
    ]
    source_path = next((candidate for candidate in input_candidates if candidate.exists()), None)
    if source_path is None:
        if logger:
            logger.warning("enrich.linkedin skipped: no dataset available")
        return {"status": "SKIPPED", "reason": "NO_INPUT_DATA"}

    try:
        if source_path.suffix == ".parquet":
            df_in = pd.read_parquet(source_path)
        else:
            df_in = pd.read_csv(source_path)
    except Exception as exc:  # pragma: no cover - defensive
        if logger:
            logger.exception("Failed to load input data for enrich.linkedin from %s", source_path)
        return {"status": "FAIL", "error": str(exc)}

    if df_in.empty:
        if logger:
            logger.info("enrich.linkedin skipped: empty dataset (%s)", source_path.name)
        return {"status": "SKIPPED", "reason": "EMPTY_INPUT"}

    linkedin_cfg = (ctx.get("enrichment_config") or {}).get("linkedin") or {}
    df_out, summary = process_linkedin(df_in, linkedin_cfg)

    output_path = outdir / "linkedin_enriched.parquet"
    csv_path = outdir / "linkedin_enriched.csv"
    df_out.to_parquet(output_path, index=False)
    df_out.to_csv(csv_path, index=False)

    duration = round(time.time() - start, 3)
    if logger:
        logger.info(
            "enrich.linkedin completed: matches=%d (duration=%.3fs)",
            summary.get("matches", 0),
            duration,
        )

    return {
        "status": "OK",
        "file": str(output_path),
        "rows": len(df_out),
        "matches": summary.get("matches", 0),
        "duration_s": duration,
        "source": str(source_path),
    }


__all__ = ["process_linkedin", "run"]
