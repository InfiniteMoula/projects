from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin, urlparse
import pandas as pd
import phonenumbers
from bs4 import BeautifulSoup
from phonenumbers import PhoneNumberType
from phonenumbers.phonenumberutil import NumberParseException

try:
    import tldextract
except ImportError:  # pragma: no cover - defensive
    tldextract = None

from net.http_client import HttpClient
from net.http_client import RequestLimiter
from ai import extract_contacts as extract_contacts_with_llm
from constants import GENERIC_EMAIL_DOMAINS as BASE_GENERIC_EMAIL_DOMAINS
from constants import GENERIC_EMAIL_PREFIXES as BASE_GENERIC_EMAIL_PREFIXES
from utils import scoring
from utils.email_validation import has_mx_record
from net import robots, sitemap as sitemap_utils

LOGGER = logging.getLogger("enrich.enrich_contacts")
NumberType = PhoneNumberType

EMAIL_REGEX = re.compile(
    r"(?<![A-Z0-9._%+-])(?:mailto:)?([A-Z0-9][A-Z0-9._%+-]{0,63}@[A-Z0-9.-]+\.[A-Z]{2,63})",
    re.IGNORECASE,
)
EMAIL_OBFUSCATIONS: Tuple[Tuple[str, str], ...] = (
    (r"\s*\[\s*at\s*\]\s*", "@"),
    (r"\s*\(\s*at\s*\)\s*", "@"),
    (r"\sarobase\s", "@"),
    (r"\s*\[\s*arrobase\s*\]\s*", "@"),
    (r"\s*\(\s*arrobase\s*\)\s*", "@"),
    (r"\s*(?:\bat\b|\[at\]|\(at\))\s*", "@"),
    (r"\s*\[\s*dot\s*\]\s*", "."),
    (r"\s*\(\s*dot\s*\)\s*", "."),
    (r"\s*(?:\bdot\b|\bpoint\b)\s*", "."),
    (r"\s*\[\s*point\s*\]\s*", "."),
    (r"\s*\(\s*point\s*\)\s*", "."),
    (r"[\[\]\(\)]", " "),
)
GENERIC_EMAIL_PREFIXES: Set[str] = set(BASE_GENERIC_EMAIL_PREFIXES)
LINKEDIN_PATTERN = re.compile(
    r"https?://(?:[a-z]{2,3}\.)?linkedin\.com/(?:company|in)/[A-Za-z0-9_\-/%]+",
    re.IGNORECASE,
)

DEFAULT_PATHS = (
    "/contact",
    "/contacts",
    "/nous-contacter",
    "/nous_contacter",
    "/contactez-nous",
    "/contactez_nous",
    "/mentions-legales",
    "/mentions_legales",
    "/mentions",
    "/a-propos",
    "/a_propos",
    "/about",
    "/about-us",
    "/privacy",
    "/politique-de-confidentialite",
    "/politique_confidentialite",
    "/rgpd",
)
DEFAULT_MAX_PAGES = 8
DEFAULT_SITEMAP_LIMIT = 5
DEFAULT_TIMEOUT = 8.0
DEFAULT_CHUNK_SIZE = 300
DEFAULT_MAX_WORKERS = 8
MX_RECORD_PENALTY = 0.4


@dataclass
class EmailCandidate:
    value: str
    source_url: str
    page_type: str
    is_nominative: bool
    on_company_domain: bool
    score: float = 0.0


@dataclass
class PhoneCandidate:
    value: str
    source_url: str
    page_type: str
    number_type: NumberType
    score: float = 0.0
    city_match: bool = False


def _normalize_items(values: object) -> List[str]:
    if not values:
        return []
    if isinstance(values, (str, bytes)):
        candidates = [values]
    elif isinstance(values, Mapping):
        candidates = list(values.values())
    elif isinstance(values, Iterable):
        candidates = list(values)
    else:
        candidates = [values]

    normalized: List[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip().lower()
        if text:
            normalized.append(text)
    return normalized


def _configure_generic_email_filters(cfg: Mapping[str, Any]) -> None:
    prefixes = {item for item in BASE_GENERIC_EMAIL_PREFIXES if item}
    prefixes.update(_normalize_items(cfg.get("email_generic_prefixes")))

    GENERIC_EMAIL_PREFIXES.clear()
    GENERIC_EMAIL_PREFIXES.update(prefixes)

    domains = {item for item in BASE_GENERIC_EMAIL_DOMAINS if item}
    domains.update(_normalize_items(cfg.get("email_generic_domains")))

    scoring.set_generic_email_filters(domains=domains, prefixes=prefixes)


def _resolve_chunk_size(cfg: Mapping[str, Any]) -> int:
    value = cfg.get("chunk_size")
    if value is None:
        return DEFAULT_CHUNK_SIZE
    try:
        size = int(value)
    except (TypeError, ValueError):
        return DEFAULT_CHUNK_SIZE
    if size <= 0:
        return DEFAULT_CHUNK_SIZE
    return max(200, min(size, 500))


def _resolve_max_workers(cfg: Mapping[str, Any]) -> int:
    value = cfg.get("max_workers")
    if value is None:
        return DEFAULT_MAX_WORKERS
    try:
        workers = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_WORKERS
    return workers if workers > 0 else DEFAULT_MAX_WORKERS


def _prepare_http_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    http_cfg_raw = cfg.get("http_client", {"timeout": DEFAULT_TIMEOUT})
    if hasattr(http_cfg_raw, "model_dump"):
        http_cfg = dict(http_cfg_raw.model_dump())
    elif hasattr(http_cfg_raw, "dict"):
        http_cfg = dict(http_cfg_raw.dict())
    elif isinstance(http_cfg_raw, Mapping):
        http_cfg = dict(http_cfg_raw)
    else:
        http_cfg = dict(http_cfg_raw or {})
    http_cfg.setdefault("timeout", DEFAULT_TIMEOUT)
    return http_cfg


def _iter_chunks(df: pd.DataFrame, chunk_size: int) -> Iterable[Tuple[int, pd.DataFrame]]:
    total = len(df)
    if chunk_size <= 0 or chunk_size >= total:
        yield 0, df
        return
    for chunk_index, start in enumerate(range(0, total, chunk_size)):
        stop = min(start + chunk_size, total)
        yield chunk_index, df.iloc[start:stop].copy()


def process_contacts(df_in: pd.DataFrame, cfg: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enrich dataframe with official contact emails and phone numbers discovered from websites
    using chunked concurrent processing.
    """

    http_cfg = _prepare_http_cfg(cfg)
    chunk_size = _resolve_chunk_size(cfg)
    max_workers = _resolve_max_workers(cfg)
    max_concurrent_requests = max(1, int(http_cfg.get("max_concurrent_requests", max_workers)))
    request_limiter = RequestLimiter(max_concurrent_requests)
    http_cfg["shared_request_limiter"] = request_limiter

    base_cfg: Dict[str, Any] = dict(cfg)
    base_cfg["http_client"] = http_cfg

    if df_in is None:
        return _process_contacts_serial(df_in, base_cfg)

    total_rows = len(df_in)
    if total_rows == 0 or max_workers <= 1 or total_rows <= chunk_size:
        return _process_contacts_serial(df_in, base_cfg)

    worker_count = min(max_workers, max(1, (total_rows + chunk_size - 1) // chunk_size))
    chunk_results: Dict[int, pd.DataFrame] = {}
    chunk_summaries: Dict[int, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {}
        for chunk_index, chunk_df in _iter_chunks(df_in, chunk_size):
            chunk_cfg = dict(base_cfg)
            chunk_cfg["http_client"] = dict(http_cfg)
            futures[executor.submit(_process_contacts_serial, chunk_df, chunk_cfg)] = chunk_index
        for future in as_completed(futures):
            chunk_index = futures[future]
            chunk_df, summary = future.result()
            chunk_results[chunk_index] = chunk_df
            chunk_summaries[chunk_index] = summary

    ordered_dataframes = [chunk_results[idx] for idx in sorted(chunk_results)]
    df_out = pd.concat(ordered_dataframes)
    df_out = df_out.reindex(df_in.index)

    total_sites = sum(summary.get("sites", 0) for summary in chunk_summaries.values())
    total_emails = sum(summary.get("emails_found", 0) for summary in chunk_summaries.values())
    total_phones = sum(summary.get("phones_found", 0) for summary in chunk_summaries.values())
    total_pages = sum(summary.get("pages_fetched", 0) for summary in chunk_summaries.values())
    weighted_time = sum(summary.get("avg_time", 0.0) * summary.get("sites", 0) for summary in chunk_summaries.values())
    avg_time = (weighted_time / total_sites) if total_sites else 0.0

    summary = {
        "sites": total_sites,
        "emails_found": total_emails,
        "phones_found": total_phones,
        "pages_fetched": total_pages,
        "avg_time": avg_time,
    }

    return df_out, summary


def _process_contacts_serial(df_in: pd.DataFrame, cfg: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enrich dataframe with official contact emails and phone numbers discovered from websites.
    """

    if df_in is None or df_in.empty:
        df_out = df_in.copy() if df_in is not None else pd.DataFrame()
        for col in (
            "email",
            "email_source",
            "email_score",
            "telephone",
            "telephone_source",
            "telephone_score",
        ):
            if col not in df_out.columns:
                df_out[col] = pd.NA
        return df_out, {
            "sites": 0,
            "emails_found": 0,
            "phones_found": 0,
            "pages_fetched": 0,
            "avg_time": 0.0,
        }

    http_cfg_raw = cfg.get("http_client", {"timeout": DEFAULT_TIMEOUT})
    if hasattr(http_cfg_raw, "model_dump"):
        http_cfg = dict(http_cfg_raw.model_dump())
    elif hasattr(http_cfg_raw, "dict"):
        http_cfg = dict(http_cfg_raw.dict())
    elif isinstance(http_cfg_raw, Mapping):
        http_cfg = dict(http_cfg_raw)
    else:
        http_cfg = dict(http_cfg_raw or {})
    _configure_generic_email_filters(cfg)
    use_sitemaps = bool(cfg.get("use_sitemap", True))
    use_robots = bool(cfg.get("use_robots", True))
    validate_mx = bool(cfg.get("validate_mx"))
    ai_fallback_enabled = bool(
        cfg.get("_ai_fallback_enabled") or cfg.get("ai_fallback_extraction")
    )
    page_paths = tuple(cfg.get("paths", DEFAULT_PATHS))
    max_pages = int(cfg.get("max_pages_per_site", DEFAULT_MAX_PAGES))
    sitemap_limit = int(cfg.get("sitemap_limit", DEFAULT_SITEMAP_LIMIT))

    if not use_robots:
        http_cfg["respect_robots"] = False

    http_client = HttpClient(http_cfg)
    robots_user_agent = http_cfg.get("default_headers", {}).get("User-Agent")
    if not robots_user_agent:
        robots_user_agent = http_client.pick_user_agent()
    if use_robots or use_sitemaps:
        robots.configure(http_client, cache_ttl=http_cfg.get("robots_cache_ttl"))
    total_sites = 0
    total_emails = 0
    total_phones = 0
    total_pages = 0
    timings: List[float] = []
    mx_cache: Dict[str, bool] = {}

    df_out = df_in.copy()
    for col in (
        "email",
        "email_source",
        "email_score",
        "telephone",
        "telephone_source",
        "telephone_score",
    ):
        if col not in df_out.columns:
            df_out[col] = pd.NA

    try:
        for row in df_out.itertuples(index=True):
            idx = row.Index
            site_raw = _safe_str(getattr(row, "site_web", ""))
            if not site_raw:
                continue
            base_url = _normalize_site_url(site_raw)
            if not base_url:
                continue

            total_sites += 1
            start = time.perf_counter()
            try:
                company_domain = _extract_domain(base_url)
            except ValueError:
                company_domain = ""
            norm_city = _normalize_text(_safe_str(getattr(row, "ville", "")))
            city_code = (
                _safe_str(getattr(row, "departement", ""))
                or _safe_str(getattr(row, "code_postal", ""))
                or _safe_str(getattr(row, "cp", ""))
            )

            candidate_urls = _discover_candidate_urls(
                http_client,
                base_url,
                page_paths=page_paths,
                sitemap_limit=sitemap_limit,
                max_pages=max_pages,
                use_sitemaps=use_sitemaps,
                use_robots=use_robots,
                robots_user_agent=robots_user_agent if use_robots else None,
            )
            if not candidate_urls:
                continue

            # Enforce limit
            candidates_list = list(candidate_urls)[:max_pages]
            fetched_pages = _fetch_pages(http_client, candidates_list)

            email_candidates: List[EmailCandidate] = []
            phone_candidates: List[PhoneCandidate] = []
            linkedin_links: Set[str] = set()

            for url, (status, body) in fetched_pages.items():
                if status != 200 or not body:
                    continue
                total_pages += 1
                page_type = _classify_page(url, base_url)
                emails, phones, linkedins = _extract_contacts_from_html(
                    url=url,
                    html=body,
                    company_domain=company_domain,
                    norm_city=norm_city,
                    fallback_enabled=ai_fallback_enabled,
                    fallback_hints={"url": url},
                )
                linkedin_links.update(linkedins)

                for email, on_domain, is_nominative in emails:
                    candidate = EmailCandidate(
                        value=email,
                        source_url=url,
                        page_type=page_type,
                        is_nominative=is_nominative,
                        on_company_domain=on_domain,
                    )
                    candidate.score = scoring.score_email(candidate.value, company_domain)
                    if validate_mx and "@" in candidate.value:
                        domain = candidate.value.rsplit("@", 1)[-1].strip().lower()
                        if domain:
                            has_mx = mx_cache.get(domain)
                            if has_mx is None:
                                has_mx = has_mx_record(domain)
                                mx_cache[domain] = has_mx
                            if not has_mx:
                                candidate.score = max(0.0, candidate.score - MX_RECORD_PENALTY)
                    email_candidates.append(candidate)

                for phone_value, number_type, city_match in phones:
                    candidate = PhoneCandidate(
                        value=phone_value,
                        source_url=url,
                        page_type=page_type,
                        number_type=number_type,
                        city_match=city_match,
                    )
                    candidate.score = scoring.score_phone(candidate.value, city_code)
                    phone_candidates.append(candidate)

            best_email = _select_best_email(email_candidates)
            best_phone = _select_best_phone(phone_candidates)

            if best_email:
                total_emails += 1
                df_out.at[idx, "email"] = best_email.value
                df_out.at[idx, "email_source"] = best_email.source_url
                df_out.at[idx, "email_score"] = round(best_email.score, 3)

            if best_phone:
                total_phones += 1
                df_out.at[idx, "telephone"] = best_phone.value
                df_out.at[idx, "telephone_source"] = best_phone.source_url
                df_out.at[idx, "telephone_score"] = round(best_phone.score, 3)

            duration = time.perf_counter() - start
            timings.append(duration)
            LOGGER.info(
                "contacts search site=%s emails=%d phones=%d linkedin=%d pages=%d duration=%.3fs",
                base_url,
                len(email_candidates),
                len(phone_candidates),
                len(linkedin_links),
                len(fetched_pages),
                duration,
            )
    finally:
        http_client.close()
        try:
            asyncio.run(http_client.aclose())
        except RuntimeError:
            # already running loop; ignore
            pass

    avg_time = (sum(timings) / len(timings)) if timings else 0.0
    LOGGER.info(
        "contacts enrichment summary sites=%d emails_found=%d phones_found=%d pages=%d avg_time=%.3fs",
        total_sites,
        total_emails,
        total_phones,
        total_pages,
        avg_time,
    )
    summary = {
        "sites": total_sites,
        "emails_found": total_emails,
        "phones_found": total_phones,
        "pages_fetched": total_pages,
        "avg_time": avg_time,
    }
    return df_out, summary


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_site_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    parsed = urlparse(url, scheme="https")
    if not parsed.netloc:
        parsed = urlparse(f"https://{url}")
    if not parsed.netloc:
        return ""
    scheme = parsed.scheme or "https"
    if scheme not in {"http", "https"}:
        scheme = "https"
    normalized = parsed._replace(scheme=scheme, fragment="")
    return normalized.geturl()


def _discover_candidate_urls(
    http: HttpClient,
    base_url: str,
    *,
    page_paths: Sequence[str],
    sitemap_limit: int,
    max_pages: int,
    use_sitemaps: bool,
    use_robots: bool,
    robots_user_agent: Optional[str],
) -> List[str]:
    parsed = urlparse(base_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    urls: List[str] = []
    seen: Set[str] = set()

    def is_allowed(target: str) -> bool:
        if not use_robots or not robots_user_agent:
            return True
        try:
            return robots.is_allowed(target, robots_user_agent)
        except Exception:  # pragma: no cover - defensive
            return True

    def add_url(path_or_url: str) -> bool:
        if len(urls) >= max_pages:
            return False
        target = urljoin(root + "/", path_or_url)
        if urlparse(target).netloc != parsed.netloc:
            return False
        if target in seen:
            return False
        if not is_allowed(target):
            return False
        seen.add(target)
        urls.append(target)
        return True

    add_url(base_url)
    for path in page_paths:
        add_url(path)

    if use_sitemaps and sitemap_limit != 0:
        sitemap_candidates = sitemap_utils.discover_sitemap_urls(base_url, http)
        limit = sitemap_limit if sitemap_limit > 0 else None
        added = 0
        for link in sitemap_candidates:
            if limit is not None and added >= limit:
                break
            if add_url(link):
                added += 1

    return urls


def _fetch_pages(http: HttpClient, urls: Sequence[str]) -> Dict[str, Tuple[int, str]]:
    if not urls:
        return {}

    async def _runner() -> Dict[str, Tuple[int, str]]:
        return await http.fetch_all(list(urls))

    return asyncio.run(_runner())


def _classify_page(url: str, base_url: str) -> str:
    path = urlparse(url).path.lower()
    if path in {"/", ""} or url.rstrip("/") == base_url.rstrip("/"):
        return "home"
    if "contact" in path:
        return "contact"
    if "mention" in path:
        return "legal"
    if "privacy" in path or "confidential" in path or "rgpd" in path:
        return "privacy"
    if "about" in path or "apropos" in path or "a-propos" in path:
        return "about"
    return "other"


def _extract_contacts_from_html(
    *,
    url: str,
    html: str,
    company_domain: str,
    norm_city: str,
    fallback_enabled: bool = False,
    fallback_hints: Optional[Mapping[str, object]] = None,
) -> Tuple[List[Tuple[str, bool, bool]], List[Tuple[str, NumberType, bool]], Set[str]]:
    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text_content = soup.get_text(" ", strip=True)
        href_text = " ".join(a.get("href", "") for a in soup.find_all("a"))
        combined = f"{text_content} {href_text}"
        deobfuscated = _deobfuscate(combined)

        emails = _extract_emails(deobfuscated, company_domain)
        phones = _extract_phones(deobfuscated, norm_city)
        linkedins = set(match.group(0) for match in LINKEDIN_PATTERN.finditer(combined))
        return emails, phones, linkedins
    except Exception as exc:
        if not fallback_enabled:
            raise

        hints: Dict[str, object] = {
            "url": url,
            "company_domain": company_domain,
            "city": norm_city,
            "error": str(exc),
        }
        if fallback_hints:
            hints.update(fallback_hints)

        LOGGER.warning(
            "HTML parsing failed for %s; attempting AI fallback", url, exc_info=LOGGER.isEnabledFor(logging.DEBUG)
        )

        try:
            payload = extract_contacts_with_llm(html, hints)
        except Exception as fallback_exc:  # pragma: no cover - defensive
            LOGGER.error(
                "AI fallback extraction failed for %s: %s", url, fallback_exc
            )
            raise exc

        return _normalise_llm_contacts(payload, company_domain, norm_city)


def _deobfuscate(text: str) -> str:
    result = text
    for pattern, replacement in EMAIL_OBFUSCATIONS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = result.replace("[at]", "@").replace("(at)", "@")
    return result


def _normalise_llm_contacts(
    payload: Mapping[str, object] | None,
    company_domain: str,
    norm_city: str,
) -> Tuple[List[Tuple[str, bool, bool]], List[Tuple[str, NumberType, bool]], Set[str]]:
    if not isinstance(payload, Mapping):
        return [], [], set()

    email_candidates: List[Tuple[str, bool, bool]] = []
    for raw_email in _normalize_items(payload.get("emails")):
        sanitized = _sanitize_email(str(raw_email))
        if not _is_probable_email(sanitized):
            continue
        local, _, domain = sanitized.partition("@")
        on_domain = bool(company_domain and domain.endswith(company_domain))
        email_candidates.append((sanitized, on_domain, _is_nominative_local(local)))

    phone_inputs = " ".join(_normalize_items(payload.get("phones")))
    phone_candidates = _extract_phones(phone_inputs, norm_city) if phone_inputs else []

    linkedin_raw = payload.get("linkedin")
    if not linkedin_raw:
        linkedin_raw = payload.get("linkedins")
    linkedin_candidates = {
        str(item).strip()
        for item in _normalize_items(linkedin_raw)
        if str(item).strip()
    }

    return email_candidates, phone_candidates, linkedin_candidates


def _extract_emails(text: str, company_domain: str) -> List[Tuple[str, bool, bool]]:
    candidates: Set[str] = set()
    normalized_text = text
    for pattern, replacement in EMAIL_OBFUSCATIONS:
        normalized_text = re.sub(pattern, replacement, normalized_text, flags=re.IGNORECASE)
    for match in EMAIL_REGEX.finditer(normalized_text):
        email = match.group(1).strip().lower()
        cleaned = _sanitize_email(email)
        if not _is_probable_email(cleaned):
            continue
        candidates.add(cleaned)

    results: List[Tuple[str, bool, bool]] = []
    for email in candidates:
        local, _, domain = email.partition("@")
        on_domain = bool(company_domain and domain.endswith(company_domain))
        is_nominative = _is_nominative_local(local)
        results.append((email, on_domain, is_nominative))
    return results


def _sanitize_email(email: str) -> str:
    if not email:
        return ""
    cleaned = email
    for pattern, replacement in EMAIL_OBFUSCATIONS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(".,;:<>[](){} ").replace("..", ".")
    cleaned = re.sub(r"\s*@\s*", "@", cleaned)
    cleaned = re.sub(r"\s*\.\s*", ".", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.lower()


def _is_probable_email(email: str) -> bool:
    if not email or "@" not in email:
        return False
    local, _, domain = email.partition("@")
    if not local or not domain or "." not in domain:
        return False
    tld = domain.rsplit(".", 1)[-1].lower()
    if len(tld) < 2:
        return False
    if any(ext in tld for ext in ("png", "jpg", "jpeg", "gif", "svg", "webp", "js", "css")):
        return False
    return True


def _is_nominative_local(local: str) -> bool:
    lowered = local.lower()
    if lowered in GENERIC_EMAIL_PREFIXES:
        return False
    return bool(re.search(r"[.-]", lowered)) or lowered not in GENERIC_EMAIL_PREFIXES


def _extract_phones(text: str, norm_city: str) -> List[Tuple[str, NumberType, bool]]:
    phones: Dict[str, Tuple[NumberType, bool]] = {}
    for match in phonenumbers.PhoneNumberMatcher(text, "FR"):
        try:
            number = phonenumbers.parse(match.raw_string, "FR")
        except NumberParseException:
            continue
        if not phonenumbers.is_valid_number(number):
            continue
        formatted = phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)
        num_type = phonenumbers.number_type(number)
        city_match = False
        if norm_city:
            snippet = _normalize_text(match.raw_string)
            city_match = norm_city in snippet
        existing = phones.get(formatted)
        if existing:
            prev_type, prev_city = existing
            if prev_type == num_type and prev_city:
                continue
            if prev_city and not city_match:
                continue
        phones[formatted] = (num_type, city_match or (existing[1] if existing else False))
    return [(phone, meta[0], meta[1]) for phone, meta in phones.items()]


def _select_best_email(candidates: Iterable[EmailCandidate]) -> Optional[EmailCandidate]:
    best: Optional[EmailCandidate] = None
    for candidate in candidates:
        if not best or candidate.score > best.score or (
            candidate.score == best.score and candidate.page_type == "contact" and best.page_type != "contact"
        ):
            best = candidate
    return best


def _select_best_phone(candidates: Iterable[PhoneCandidate]) -> Optional[PhoneCandidate]:
    best: Optional[PhoneCandidate] = None
    for candidate in candidates:
        if (
            best is None
            or candidate.score > best.score
            or (
                candidate.score == best.score
                and best is not None
                and candidate.number_type == NumberType.FIXED_LINE
                and best.number_type != NumberType.FIXED_LINE
            )
            or (
                candidate.score == best.score
                and best is not None
                and candidate.number_type == best.number_type
                and candidate.city_match
                and not best.city_match
            )
        ):
            best = candidate
    return best


def _extract_domain(url: str) -> str:
    if not tldextract:
        raise ValueError("tldextract is required for domain extraction")
    parts = tldextract.extract(url)
    if parts.domain and parts.suffix:
        return f"{parts.domain}.{parts.suffix}".lower()
    return (urlparse(url).netloc or "").lower()


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


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
            logger.warning("enrich.contacts skipped: no input dataset found")
        return {"status": "SKIPPED", "reason": "NO_INPUT_DATA"}

    try:
        if source_path.suffix == ".parquet":
            df_in = pd.read_parquet(source_path)
        else:
            df_in = pd.read_csv(source_path)
    except Exception as exc:  # pragma: no cover - defensive
        if logger:
            logger.exception("Failed to load input data for enrich.contacts from %s", source_path)
        return {"status": "FAIL", "error": str(exc)}

    if df_in.empty:
        if logger:
            logger.info("enrich.contacts skipped: empty dataset (%s)", source_path.name)
        return {"status": "SKIPPED", "reason": "EMPTY_INPUT"}

    enrichment_cfg = ctx.get("enrichment_config") or {}
    contacts_cfg_raw = (
        enrichment_cfg.get("contacts") if isinstance(enrichment_cfg, Mapping) else getattr(enrichment_cfg, "contacts", {})
    ) or {}
    if hasattr(contacts_cfg_raw, "model_dump"):
        contacts_cfg = dict(contacts_cfg_raw.model_dump())
    elif hasattr(contacts_cfg_raw, "dict"):
        contacts_cfg = dict(contacts_cfg_raw.dict())
    elif isinstance(contacts_cfg_raw, Mapping):
        contacts_cfg = dict(contacts_cfg_raw)
    else:
        contacts_cfg = dict(contacts_cfg_raw or {})

    ai_cfg_raw = (
        enrichment_cfg.get("ai")
        if isinstance(enrichment_cfg, Mapping)
        else getattr(enrichment_cfg, "ai", None)
    )
    ai_cfg_dict: Dict[str, Any] = {}
    if ai_cfg_raw is not None:
        if hasattr(ai_cfg_raw, "model_dump"):
            ai_cfg_dict = dict(ai_cfg_raw.model_dump())
        elif hasattr(ai_cfg_raw, "dict"):
            ai_cfg_dict = dict(ai_cfg_raw.dict())
        elif isinstance(ai_cfg_raw, Mapping):
            ai_cfg_dict = dict(ai_cfg_raw)
    if ai_cfg_dict.get("fallback_extraction"):
        contacts_cfg["_ai_fallback_enabled"] = True

    if isinstance(cfg, Mapping):
        overrides = {}
        for key in ("chunk_size", "max_workers"):
            value = cfg.get(key)
            if value is not None:
                overrides[key] = value
        if overrides:
            contacts_cfg.update(overrides)

    df_out, summary = process_contacts(df_in, contacts_cfg)

    output_path = outdir / "contacts_enriched.parquet"
    csv_path = outdir / "contacts_enriched.csv"
    df_out.to_parquet(output_path, index=False)
    df_out.to_csv(csv_path, index=False)

    duration = round(time.time() - start, 3)
    if logger:
        logger.info(
            "enrich.contacts completed: emails=%d phones=%d (duration=%.3fs)",
            summary.get("emails_found", 0),
            summary.get("phones_found", 0),
            duration,
        )

    return {
        "status": "OK",
        "file": str(output_path),
        "rows": len(df_out),
        "emails_found": summary.get("emails_found", 0),
        "phones_found": summary.get("phones_found", 0),
        "duration_s": duration,
        "source": str(source_path),
    }


__all__ = ["process_contacts", "run"]
