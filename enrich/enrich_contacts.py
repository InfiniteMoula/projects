from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import pandas as pd
import phonenumbers
from bs4 import BeautifulSoup
from phonenumbers.phonenumberutil import NumberParseException, NumberType

try:
    import tldextract
except ImportError:  # pragma: no cover - defensive
    tldextract = None

from net.http_client import HttpClient

LOGGER = logging.getLogger("enrich.enrich_contacts")

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
GENERIC_EMAIL_PREFIXES = {
    "contact",
    "info",
    "hello",
    "support",
    "service",
    "commercial",
    "vente",
    "admin",
    "administration",
    "compta",
    "billing",
    "facturation",
    "direction",
    "rh",
    "recrutement",
    "postmaster",
    "noreply",
    "no-reply",
    "bonjour",
}
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
SITEMAP_KEYWORDS = (
    "contact",
    "mention",
    "legal",
    "privacy",
    "confidential",
    "about",
    "nous",
    "rgpd",
    "coord",
    "contactez",
)

DEFAULT_MAX_PAGES = 8
DEFAULT_SITEMAP_LIMIT = 5
DEFAULT_TIMEOUT = 8.0


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


def run(df_in: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
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
        return df_out

    http_cfg = cfg.get("http_client", {"timeout": DEFAULT_TIMEOUT})
    page_paths = tuple(cfg.get("paths", DEFAULT_PATHS))
    max_pages = int(cfg.get("max_pages_per_site", DEFAULT_MAX_PAGES))
    sitemap_limit = int(cfg.get("sitemap_limit", DEFAULT_SITEMAP_LIMIT))

    http_client = HttpClient(http_cfg)
    total_sites = 0
    total_emails = 0
    total_phones = 0
    total_pages = 0
    timings: List[float] = []

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

            candidate_urls = _discover_candidate_urls(
                http_client,
                base_url,
                page_paths=page_paths,
                sitemap_limit=sitemap_limit,
                max_pages=max_pages,
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
                    candidate.score = _score_email(candidate, company_domain, page_type)
                    email_candidates.append(candidate)

                for phone_value, number_type, city_match in phones:
                    candidate = PhoneCandidate(
                        value=phone_value,
                        source_url=url,
                        page_type=page_type,
                        number_type=number_type,
                        city_match=city_match,
                    )
                    candidate.score = _score_phone(candidate)
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
    return df_out


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
) -> List[str]:
    parsed = urlparse(base_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    urls: List[str] = []
    seen: Set[str] = set()

    def add_url(path_or_url: str) -> None:
        if len(urls) >= max_pages:
            return
        target = urljoin(root + "/", path_or_url)
        if urlparse(target).netloc != parsed.netloc:
            return
        if target not in seen:
            seen.add(target)
            urls.append(target)

    add_url(base_url)
    for path in page_paths:
        add_url(path)

    for sitemap_url in _discover_sitemaps(http, root):
        try:
            response = http.get(sitemap_url)
        except Exception:  # pragma: no cover - defensive
            continue
        if response.status_code != 200 or not response.text:
            continue
        for link in _extract_sitemap_links(response.text, sitemap_limit):
            add_url(link)

    return urls


def _discover_sitemaps(http: HttpClient, root: str) -> Set[str]:
    parsed = urlparse(root)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    sitemaps: Set[str] = set()

    try:
        robots_resp = http.get(robots_url)
    except Exception:  # pragma: no cover - defensive
        robots_resp = None

    if robots_resp and robots_resp.status_code == 200 and robots_resp.text:
        for line in robots_resp.text.splitlines():
            if line.lower().startswith("sitemap:"):
                sitemap_url = line.split(":", 1)[1].strip()
                if sitemap_url:
                    sitemaps.add(sitemap_url)

    default_sitemap = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
    sitemaps.add(default_sitemap)
    return sitemaps


def _extract_sitemap_links(xml_text: str, limit: int) -> List[str]:
    links: List[str] = []
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return links

    for loc in root.iter("{*}loc"):
        if not loc.text:
            continue
        href = loc.text.strip()
        if any(keyword in href.lower() for keyword in SITEMAP_KEYWORDS):
            links.append(href)
        if len(links) >= limit:
            break
    return links


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
) -> Tuple[List[Tuple[str, bool, bool]], List[Tuple[str, NumberType, bool]], Set[str]]:
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


def _deobfuscate(text: str) -> str:
    result = text
    for pattern, replacement in EMAIL_OBFUSCATIONS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = result.replace("[at]", "@").replace("(at)", "@")
    return result


def _extract_emails(text: str, company_domain: str) -> List[Tuple[str, bool, bool]]:
    candidates: Set[str] = set()
    for match in EMAIL_REGEX.finditer(text):
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
    email = email.strip(".,;:<>[](){} ").replace("..", ".")
    email = re.sub(r"\s*@\s*", "@", email)
    email = re.sub(r"\s*\.\s*", ".", email)
    return email.lower()


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


def _score_email(candidate: EmailCandidate, company_domain: str, page_type: str) -> float:
    local, _, domain = candidate.value.partition("@")
    score = 0.2
    similarity = 0.0
    if company_domain:
        try:
            from rapidfuzz import fuzz

            similarity = fuzz.partial_ratio(company_domain, domain) / 100.0
        except Exception:
            similarity = 0.0
    score += 0.2 * similarity
    if candidate.on_company_domain:
        score += 0.4
    elif company_domain and domain.split(".")[-1] == company_domain.split(".")[-1]:
        score += 0.2
    if candidate.is_nominative and local not in GENERIC_EMAIL_PREFIXES:
        score += 0.2
    else:
        score += 0.05
    if page_type == "contact":
        score += 0.2
    elif page_type in {"legal", "privacy", "about"}:
        score += 0.1
    return min(score, 1.0)


def _score_phone(candidate: PhoneCandidate) -> float:
    score = 0.2
    if candidate.number_type == NumberType.FIXED_LINE:
        score += 0.4
    elif candidate.number_type == NumberType.MOBILE:
        score += 0.3
    else:
        score += 0.1
    if candidate.page_type == "contact":
        score += 0.2
    elif candidate.page_type in {"legal", "privacy", "about"}:
        score += 0.1
    if candidate.city_match:
        score += 0.1
    return min(score, 1.0)


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
        if not best or candidate.score > best.score or (
            candidate.score == best.score
            and candidate.number_type == NumberType.FIXED_LINE
            and (best.number_type != NumberType.FIXED_LINE)
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


__all__ = ["run"]
