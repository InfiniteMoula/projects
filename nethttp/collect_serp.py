"""Collect official site candidates from public SERP results."""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from utils import budget_middleware, io
from utils.http import HttpError, request_with_backoff
from utils.rate import PerHostRateLimiter
from utils.ua import UserAgentPool, load_user_agent_pool
from utils.url import canonicalize, hostname, registered_domain
from utils.state import SequentialRunState

BING_URL = "https://www.bing.com/search"
LEGAL_STOPWORDS = {
    "SARL",
    "S.A.R.L",
    "SAS",
    "S.A.S",
    "SASU",
    "SA",
    "S.A",
    "SCOP",
    "SCI",
    "SC",
    "EURL",
    "EARL",
    "SNC",
}
GENERIC_DOMAINS = {"pagesjaunes.fr", "societe.com", "manageo.fr", "verif.com", "infogreffe.fr", "facebook.com"}
SERP_HEADERS = {
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.6",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
SERP_MAX_RESULTS_DEFAULT = 5
SERP_TIMEOUT_DEFAULT = 8.0


@dataclass
class CompanyQuery:
    siren: Optional[str]
    denomination: str
    city: Optional[str]
    postal_code: Optional[str]
    address: Optional[str]


@dataclass
class SerpSelection:
    url: str
    domain: str
    rank: int
    confidence: float
    title: str
    snippet: str


def _normalize_name(name: str) -> str:
    cleaned = name.upper()
    for word in sorted(LEGAL_STOPWORDS, key=len, reverse=True):
        pattern = rf"\b{re.escape(word)}\b"
        cleaned = re.sub(pattern, " ", cleaned)
    cleaned = re.sub(r"[^A-Z0-9 ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def _build_queries(row: CompanyQuery) -> List[str]:
    base_parts: List[str] = []
    if row.denomination:
        base_parts.append(row.denomination.strip())
    if row.postal_code:
        base_parts.append(str(row.postal_code).strip())
    if row.city:
        base_parts.append(row.city.strip())
    base_parts.append("site officiel")
    primary = " ".join(part for part in base_parts if part)
    queries = [primary]
    if row.postal_code:
        fallback_parts = [row.denomination.strip() if row.denomination else ""]
        if row.city:
            fallback_parts.append(row.city.strip())
        fallback_parts.append("site officiel")
        fallback = " ".join(part for part in fallback_parts if part)
        if fallback and fallback != primary:
            queries.append(fallback)
    if row.address:
        address_query = f"{row.denomination} {row.address}".strip()
        if address_query and address_query not in queries:
            queries.append(address_query)
    return queries


def _parse_bing(html: str, max_results: int) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    results: List[Dict[str, str]] = []
    root = soup.find("ol", id="b_results")
    if not root:
        root = soup
    for li in root.select("li.b_algo"):
        link = li.find("a", href=True)
        if not link:
            continue
        href = link.get("href")
        if not href:
            continue
        url = canonicalize(href)
        if not url:
            continue
        domain = registered_domain(url)
        if not domain or domain in GENERIC_DOMAINS:
            continue
        title = link.get_text(" ", strip=True)
        snippet_tag = li.find("p")
        snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""
        rank = len(results) + 1
        results.append(
            {
                "url": url,
                "domain": domain,
                "title": title,
                "snippet": snippet,
                "rank": rank,
            }
        )
        if len(results) >= max_results:
            break
    return results


def _score_result(result: Dict[str, str], name_norm: str, city: Optional[str]) -> float:
    target = f"{result['title']} {result['snippet']} {result['domain']}"
    url_score = fuzz.partial_ratio(name_norm, result['domain'])
    title_score = fuzz.partial_ratio(name_norm, result['title'])
    mix_score = fuzz.partial_ratio(name_norm, target)
    score = max(url_score, title_score, mix_score)
    tokens = [token for token in name_norm.split() if token]
    if tokens:
        domain_clean = re.sub(r'[^A-Z0-9]', '', result['domain'].upper())
        if all(token in domain_clean for token in tokens):
            score = max(score, 90.0)
        elif any(token in domain_clean for token in tokens):
            score = max(score, score + 15)
    if city and city.lower() in target.lower():
        score += 5
    return min(score, 100.0)


def _select_best(results: List[Dict[str, str]], row: CompanyQuery) -> Optional[SerpSelection]:
    if not results:
        return None
    name_norm = _normalize_name(row.denomination)
    best: Optional[Tuple[float, Dict[str, str]]] = None
    for result in results:
        score = _score_result(result, name_norm, row.city)
        if best is None or score > best[0]:
            best = (score, result)
    if not best:
        return None
    score, result = best
    return SerpSelection(
        url=result["url"],
        domain=result["domain"],
        rank=int(result["rank"]),
        confidence=float(score),
        title=result["title"],
        snippet=result["snippet"],
    )


def _company_key(idx: int, row: pd.Series, denomination: str) -> str:
    siren = str(row.get("siren") or "").strip()
    if siren:
        return f"{siren}:{idx}"
    city = str(row.get("ville") or row.get("commune") or "").strip()
    postal_code = str(row.get("code_postal") or "").strip()
    denom_token = re.sub(r"\s+", "_", denomination.lower()) if denomination else "unknown"
    city_token = re.sub(r"\s+", "_", city.lower()) if city else "nocity"
    postal_token = postal_code or "nopostal"
    return f"{idx}:{denom_token}:{city_token}:{postal_token}"


def _iter_companies(df: pd.DataFrame) -> Iterable[Tuple[str, CompanyQuery]]:
    for idx, row in df.iterrows():
        denomination = str(row.get("denomination") or row.get("raison_sociale") or "").strip()
        if not denomination:
            continue
        key = _company_key(idx, row, denomination)
        yield key, CompanyQuery(
            siren=str(row.get("siren") or "").strip() or None,
            denomination=denomination,
            city=(str(row.get("ville") or row.get("commune") or "").strip() or None),
            postal_code=(str(row.get("code_postal") or "").strip() or None),
            address=(str(row.get("adresse") or row.get("adresse_complete") or "").strip() or None),
        )


def run(cfg: dict, ctx: dict) -> dict:
    logger = ctx.get("logger")
    outdir = Path(ctx["outdir"])
    input_path = outdir / "normalized.parquet"
    if not input_path.exists():
        if logger:
            logger.warning("collect_serp: normalized.parquet not found at %s", input_path)
        return {"status": "SKIPPED", "reason": "NO_NORMALIZED"}

    serp_cfg = (cfg.get("serp") or {})
    engine = serp_cfg.get("engine", "bing").lower()
    if engine != "bing":
        raise ValueError("Only Bing engine is supported for http.serp step")
    max_results = int(serp_cfg.get("max_results") or SERP_MAX_RESULTS_DEFAULT)
    timeout = float(ctx.get("serp_timeout_sec") or serp_cfg.get("timeout_sec") or SERP_TIMEOUT_DEFAULT)
    per_host_rps = float((serp_cfg.get("per_host_rps") or 1.0))

    df = pd.read_parquet(input_path)
    total_rows = len(df)
    sample = int(ctx.get("sample") or 0)
    if sample > 0:
        df = df.head(sample)
    elif ctx.get("dry_run"):
        df = df.head(min(10, len(df)))

    if df.empty:
        return {"status": "SKIPPED", "reason": "EMPTY"}

    companies: List[Tuple[str, CompanyQuery]] = list(_iter_companies(df))
    if not companies:
        return {"status": "SKIPPED", "reason": "NO_COMPANIES"}

    serp_dir = io.ensure_dir(outdir / "serp")
    state = SequentialRunState(serp_dir / "serp_state.json")
    state.set_metadata(total=len(companies))

    completed_map = state.metadata.get("completed_extra")
    if not isinstance(completed_map, dict):
        completed_map = {}

    ordered_keys = [key for key, _ in companies]
    pending_keys = set(state.pending(ordered_keys))

    ua_pool: UserAgentPool = ctx.get("user_agent_pool") or load_user_agent_pool(None)
    limiter = PerHostRateLimiter(per_host_rps=per_host_rps, jitter_range=(0.2, 0.8))
    request_tracker = ctx.get("request_tracker")

    queries_seen: Dict[str, List[Dict[str, str]]] = {}
    session_headers = dict(SERP_HEADERS)
    parsed_host = urlparse(BING_URL).hostname or "bing.com"
    attempts = int(state.metadata.get("attempts") or 0)

    if not pending_keys and logger:
        logger.info("collect_serp: all %d companies already processed", len(companies))

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for key, company in companies:
            if key not in pending_keys:
                continue

            state.mark_started(key)
            picked_selection: Optional[SerpSelection] = None
            picked_results: List[Dict[str, str]] = []
            used_query = ""

            try:
                for query in _build_queries(company):
                    limiter.wait(parsed_host)
                    cached = queries_seen.get(query)
                    if cached is None:
                        headers = dict(session_headers)
                        headers["User-Agent"] = ua_pool.get()
                        params = {
                            "q": query,
                            "count": max(5, max_results),
                            "mkt": "fr-FR",
                            "setLang": "fr",
                        }
                        try:
                            response = request_with_backoff(
                                client,
                                "GET",
                                BING_URL,
                                headers=headers,
                                params=params,
                                request_tracker=request_tracker,
                            )
                        except budget_middleware.BudgetExceededError:
                            state.mark_failed(key, "budget_exceeded")
                            state.set_metadata(last_error=key, attempts=attempts)
                            raise
                        except HttpError as exc:
                            if logger:
                                logger.warning("collect_serp: query '%s' failed: %s", query, exc)
                            queries_seen[query] = []
                            continue
                        if response.status_code >= 400:
                            if logger:
                                logger.debug(
                                    "collect_serp: status %s for query '%s'",
                                    response.status_code,
                                    query,
                                )
                            queries_seen[query] = []
                            continue
                        parsed_results = _parse_bing(response.text, max_results)
                        queries_seen[query] = parsed_results
                        attempts += 1
                    picked_results = queries_seen[query]
                    selection = _select_best(picked_results, company)
                    used_query = query
                    if selection:
                        picked_selection = selection
                        break

                if not picked_selection and picked_results:
                    best_raw = picked_results[0]
                    picked_selection = SerpSelection(
                        url=best_raw["url"],
                        domain=best_raw["domain"],
                        rank=int(best_raw["rank"]),
                        confidence=float(_score_result(best_raw, _normalize_name(company.denomination), company.city)),
                        title=best_raw.get("title") or "",
                        snippet=best_raw.get("snippet") or "",
                    )

                result_row = {
                    "siren": company.siren,
                    "denomination": company.denomination,
                    "ville": company.city,
                    "code_postal": company.postal_code,
                    "query": used_query,
                    "top_url": picked_selection.url if picked_selection else "",
                    "top_domain": picked_selection.domain if picked_selection else "",
                    "rank": picked_selection.rank if picked_selection else None,
                    "confidence": picked_selection.confidence if picked_selection else 0.0,
                    "title": picked_selection.title if picked_selection else "",
                    "snippet": picked_selection.snippet if picked_selection else "",
                    "results": picked_results,
                }
                state.mark_completed(key, extra=result_row)
                completed_map[key] = result_row
            except budget_middleware.BudgetExceededError:
                raise
            except Exception as exc:
                if logger:
                    logger.warning("collect_serp: failed for %s: %s", key, exc)
                fallback_row = {
                    "siren": company.siren,
                    "denomination": company.denomination,
                    "ville": company.city,
                    "code_postal": company.postal_code,
                    "query": used_query,
                    "top_url": "",
                    "top_domain": "",
                    "rank": None,
                    "confidence": 0.0,
                    "title": "",
                    "snippet": "",
                    "results": [],
                }
                state.mark_completed(key, extra=fallback_row)
                completed_map[key] = fallback_row

    out_rows: List[Dict[str, object]] = [
        completed_map[key] for key in ordered_keys if isinstance(completed_map.get(key), dict)
    ]

    output_path = serp_dir / "serp_results.parquet"
    pd.DataFrame(out_rows).to_parquet(output_path, index=False)

    stats = {
        "queries": len(out_rows),
        "attempts": attempts,
        "input_rows": int(total_rows),
    }
    state.set_metadata(attempts=attempts, last_output=str(output_path), records=len(out_rows))
    return {
        "status": "OK",
        "output": str(output_path),
        "stats": stats,
    }
