from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
import tldextract

from net.http_client import HttpClient
from constants import GENERIC_DOMAINS
from serp.providers import Result, SerpProvider, BingProvider, DuckDuckGoProvider
from serp.playwright_provider import PlaywrightBingProvider, PlaywrightGoogleProvider
from utils.scoring import score_domain

LOGGER = logging.getLogger("enrich.site_web_search")

DEFAULT_PROVIDERS = ("bing", "duckduckgo")
DEFAULT_TLDS = ("fr", "com", "eu", "net", "org")
SERP_SCORE_THRESHOLD = 0.6
HEURISTIC_SCORE_THRESHOLD = 0.5
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
    "SCOP",
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

PROVIDER_REGISTRY: Dict[str, type[SerpProvider]] = {
    "bing": BingProvider,
    "duckduckgo": DuckDuckGoProvider,
    "playwright_bing": PlaywrightBingProvider,
    "playwright_google": PlaywrightGoogleProvider,
}


@dataclass
class _Candidate:
    url: str
    source: str
    score: float
    title: str = ""


def run(df_in: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """
    Enrich a dataframe with website information discovered via SERP providers and heuristics.
    """

    if df_in is None or df_in.empty:
        df_out = df_in.copy() if df_in is not None else pd.DataFrame()
        for col in ("site_web", "site_web_source", "site_web_score"):
            if col not in df_out.columns:
                df_out[col] = pd.NA
        return df_out

    providers_cfg = cfg.get("providers", DEFAULT_PROVIDERS)
    if not providers_cfg:
        providers_cfg = DEFAULT_PROVIDERS
    raw_provider_settings = cfg.get("providers_config", {})
    provider_settings: Mapping[str, Any]
    if isinstance(raw_provider_settings, Mapping):
        provider_settings = raw_provider_settings
    else:
        provider_settings = {}
    http_cfg = cfg.get("http_client", {})
    serp_threshold = float(cfg.get("serp_score_threshold", SERP_SCORE_THRESHOLD))
    heuristic_threshold = float(cfg.get("heuristic_score_threshold", HEURISTIC_SCORE_THRESHOLD))
    heuristic_tlds = tuple(cfg.get("heuristic_tlds", DEFAULT_TLDS))
    heuristic_prefixes: Sequence[str] = tuple(cfg.get("heuristic_prefixes", ("", "www.")))

    http_client = HttpClient(http_cfg)
    generic_domains = _prepare_generic_domains(cfg.get("extra_generic_domains"))
    providers: List[Tuple[str, SerpProvider]] = []
    for provider_name in providers_cfg:
        provider_cls = PROVIDER_REGISTRY.get(str(provider_name).lower())
        if not provider_cls:
            LOGGER.warning("Unknown SERP provider %s; skipping", provider_name)
            continue
        settings = provider_settings.get(provider_name)
        if settings is None:
            settings = provider_settings.get(str(provider_name).lower(), {})
        providers.append((str(provider_name).lower(), provider_cls(settings, http_client)))

    if not providers:
        LOGGER.warning("No SERP providers configured; only heuristic lookup will run")

    df_out = df_in.copy()
    if "site_web" not in df_out.columns:
        df_out["site_web"] = pd.NA
    if "site_web_source" not in df_out.columns:
        df_out["site_web_source"] = pd.NA
    if "site_web_score" not in df_out.columns:
        df_out["site_web_score"] = pd.NA

    try:
        for row in df_out.itertuples(index=True):
            idx = row.Index
            denom_raw = _safe_str(getattr(row, "denomination", ""))
            if not denom_raw:
                continue

            city_raw = _safe_str(getattr(row, "ville", ""))
            name_for_query = _prepare_search_text(denom_raw)
            if not name_for_query:
                continue
            city_for_query = _prepare_search_text(city_raw)

            query_parts = [name_for_query]
            if city_for_query:
                query_parts.append(city_for_query)
            query_parts.append("site officiel")
            query = " ".join(query_parts)

            slug = _slugify(name_for_query)
            norm_name = _normalize_text(name_for_query)
            norm_city = _normalize_text(city_for_query)

            start = time.perf_counter()
            best_candidate: Optional[_Candidate] = None

            for provider_name, provider in providers:
                try:
                    results = provider.search(query)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Provider %s failed for %s: %s", provider_name, denom_raw, exc)
                    continue
                candidate = _pick_best_result(results, norm_name, norm_city, generic_domains)
                if not candidate:
                    continue
                candidate.source = f"SERP:{provider_name}"
                if not best_candidate or candidate.score > best_candidate.score:
                    best_candidate = candidate
                if best_candidate.score >= serp_threshold:
                    break

            if not best_candidate or best_candidate.score < serp_threshold:
                heuristic_candidate = _heuristic_lookup(
                    slug=slug,
                    http=http_client,
                    norm_name=norm_name,
                    norm_city=norm_city,
                    tlds=heuristic_tlds,
                    prefixes=heuristic_prefixes,
                )
                if heuristic_candidate and (
                    not best_candidate or heuristic_candidate.score >= best_candidate.score
                ):
                    best_candidate = heuristic_candidate

            duration = time.perf_counter() - start

            if best_candidate and best_candidate.score >= heuristic_threshold:
                site_score = score_domain(best_candidate.url, denom_raw, city_raw, best_candidate.title)
                df_out.at[idx, "site_web"] = best_candidate.url
                df_out.at[idx, "site_web_source"] = best_candidate.source
                df_out.at[idx, "site_web_score"] = site_score
                LOGGER.info(
                    "site search denomination=%s url=%s selection_score=%.3f final_score=%.3f source=%s duration=%.3fs",
                    denom_raw,
                    best_candidate.url,
                    best_candidate.score,
                    site_score,
                    best_candidate.source,
                    duration,
                )
            else:
                LOGGER.info(
                    "site search denomination=%s no_result duration=%.3fs",
                    denom_raw,
                    duration,
                )
    finally:
        try:
            http_client.close()
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug("HttpClient close failed", exc_info=True)

    return df_out


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _prepare_search_text(text: str) -> str:
    if not text:
        return ""
    stripped = _strip_accents(text).upper()
    stripped = _remove_legal_forms(stripped)
    stripped = re.sub(r"[^A-Z0-9\s-]", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.strip().lower()


def _remove_legal_forms(text: str) -> str:
    if not text:
        return ""
    forms_pattern = r"\b(" + "|".join(sorted(LEGAL_FORMS, key=len, reverse=True)) + r")\b"
    return re.sub(forms_pattern, " ", text, flags=re.IGNORECASE)


def _slugify(text: str) -> str:
    if not text:
        return ""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    slug = slug.strip("-")
    return slug


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    ascii_text = _strip_accents(text).lower()
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    ascii_text = re.sub(r"\s+", " ", ascii_text)
    return ascii_text.strip()


def _prepare_generic_domains(extra: object) -> Set[str]:
    combined: Set[str] = {domain.lower() for domain in GENERIC_DOMAINS if domain}
    if not extra:
        return combined

    if isinstance(extra, (str, bytes)):
        candidates = [extra]
    elif isinstance(extra, Mapping):
        candidates = list(extra.values())
    elif isinstance(extra, Iterable):
        candidates = list(extra)
    else:
        candidates = [extra]

    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if not text:
            continue
        normalized = _extract_registrable_domain(text)
        if normalized:
            combined.add(normalized)
    return {domain for domain in combined if domain}


def _extract_registrable_domain(value: str) -> str:
    if not value:
        return ""

    text = value.strip().lower()
    if not text:
        return ""

    if "://" in text:
        netloc = urlparse(text).netloc
    else:
        netloc = text

    netloc = netloc.split("@")[-1]
    netloc = netloc.split(":")[0]

    if tldextract:
        parts = tldextract.extract(netloc)
        if parts.domain and parts.suffix:
            return f"{parts.domain}.{parts.suffix}".lower()

    host = netloc.lstrip("www.")
    segments = [segment for segment in host.split(".") if segment]
    if len(segments) >= 2:
        return ".".join(segments[-2:]).lower()
    return host


def _is_generic_domain(domain: str, generic_domains: Set[str]) -> bool:
    if not domain:
        return False
    lowered = domain.lower()
    if lowered in generic_domains:
        return True
    return any(lowered.endswith(f".{candidate}") for candidate in generic_domains)


def _pick_best_result(
    results: Iterable[Result],
    norm_name: str,
    norm_city: str,
    generic_domains: Set[str],
) -> Optional[_Candidate]:
    best_candidate: Optional[_Candidate] = None
    for result in results:
        if not result.url:
            continue
        score = _score_serp_result(result, norm_name, norm_city, generic_domains)
        if score <= 0:
            continue
        combined_title = " ".join(part for part in (result.title, result.snippet) if part)
        candidate = _Candidate(url=result.url, source=f"SERP:{result.rank}", score=score, title=combined_title)
        if not best_candidate or candidate.score > best_candidate.score:
            best_candidate = candidate
    return best_candidate


def _score_serp_result(
    result: Result,
    norm_name: str,
    norm_city: str,
    generic_domains: Set[str],
) -> float:
    url = result.url
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return 0.0

    netloc = parsed.netloc
    registrable_domain = _extract_registrable_domain(result.domain or netloc)
    if registrable_domain and _is_generic_domain(registrable_domain, generic_domains):
        return 0.0

    domain_text = _normalize_text(result.domain or netloc)
    if not domain_text:
        return 0.0

    domain_partial = fuzz.partial_ratio(norm_name, domain_text) / 100.0
    domain_token = fuzz.token_set_ratio(norm_name, domain_text) / 100.0
    domain_score = max(domain_partial, domain_token)
    title_text = _normalize_text(result.title)
    title_score = fuzz.partial_ratio(norm_name, title_text) / 100.0 if title_text else 0.0
    city_bonus = 0.0
    if norm_city:
        snippet_text = _normalize_text(result.snippet)
        if snippet_text and norm_city in snippet_text:
            city_bonus = 0.1

    score = (0.6 * domain_score) + (0.3 * title_score) + city_bonus
    score = min(score, 1.0)
    return score


def _heuristic_lookup(
    *,
    slug: str,
    http: HttpClient,
    norm_name: str,
    norm_city: str,
    tlds: Sequence[str],
    prefixes: Sequence[str],
) -> Optional[_Candidate]:
    if not slug:
        return None

    tried: set[str] = set()

    for tld in tlds:
        for prefix in prefixes:
            hostname = f"{prefix}{slug}.{tld}".strip(".")
            if not hostname:
                continue
            if hostname in tried:
                continue
            tried.add(hostname)
            url = f"https://{hostname}"
            head_resp = http.head(url)
            if head_resp.status_code < 200 or head_resp.status_code >= 400:
                continue
            final_url = str(head_resp.url) if head_resp.url else url
            get_resp = http.get(final_url)
            if get_resp.status_code >= 400 or not get_resp.text:
                continue
            soup = BeautifulSoup(get_resp.text, "lxml")
            title_tag = soup.find("title")
            title_text = title_tag.get_text(strip=True) if title_tag else ""
            domain = tldextract.extract(final_url)
            domain_text = _normalize_text(f"{domain.domain}.{domain.suffix}") if domain.domain and domain.suffix else _normalize_text(hostname)

            domain_score = fuzz.token_sort_ratio(norm_name, domain_text) / 100.0
            title_norm = _normalize_text(title_text)
            title_score = fuzz.partial_ratio(norm_name, title_norm) / 100.0 if title_norm else 0.0
            city_bonus = 0.0
            if norm_city and title_norm and norm_city in title_norm:
                city_bonus = 0.1

            score = min(1.0, (0.65 * domain_score) + (0.25 * title_score) + city_bonus)
            if score <= 0:
                continue
            source = f"heuristic:{tld}"
            return _Candidate(url=final_url, source=source, score=score, title=title_text)
    return None


__all__ = ["run"]
