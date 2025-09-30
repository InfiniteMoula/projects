"""Parse crawled pages to extract contact details."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import phonenumbers
from bs4 import BeautifulSoup

from utils import io
from utils.url import registered_domain

EMAIL_REGEX = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
OBFUSCATION_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\s*\[\s*at\s*\]\s*", "@"),
    (r"\s*\(\s*at\s*\)\s*", "@"),
    (r"\s+arobase\s+", "@"),
    (r"\s*\[\s*arrobase\s*\]\s*", "@"),
    (r"\s*(?:\bat\b|\[at\]|\(at\))\s*", "@"),
    (r"\s*\[\s*dot\s*\]\s*", "."),
    (r"\s*\(\s*dot\s*\)\s*", "."),
    (r"\s*(?:\bdot\b|\bpoint\b)\s*", "."),
    (r"\s*\[\s*point\s*\]\s*", "."),
    (r"\s*\(\s*point\s*\)\s*", "."),
    (r"\s+\(arrobase\)\s+", "@"),
)
GENERIC_PREFIXES = {
    "contact",
    "info",
    "hello",
    "administration",
    "accueil",
    "support",
    "service",
    "compta",
    "commercial",
    "direction",
    "rh",
    "recrutement",
    "postmaster",
    "noreply",
    "no-reply",
    "bonjour",
}
SOCIAL_PATTERNS = {
    "linkedin": re.compile(r"https?://(?:[a-z]{2,3}\.)?linkedin\.com/(?:company|in|school)/[A-Za-z0-9_\-/%]+", re.IGNORECASE),
    "facebook": re.compile(r"https?://(?:www\.)?facebook\.com/[A-Za-z0-9_.\-/]+", re.IGNORECASE),
    "twitter": re.compile(r"https?://(?:www\.)?(?:twitter|x)\.com/[A-Za-z0-9_.\-/]+", re.IGNORECASE),
    "instagram": re.compile(r"https?://(?:www\.)?instagram\.com/[A-Za-z0-9_.\-/]+", re.IGNORECASE),
}
SIRET_REGEX = re.compile(r"\b(\d{14})\b")
SIREN_REGEX = re.compile(r"\b(\d{9})\b")
RCS_REGEX = re.compile(r"RCS\s+([A-Z\u00c0-\u017f\-\s]+)?\s*(\d{9})", re.IGNORECASE)
LEGAL_MANAGER_REGEX = re.compile(
    r"(g\u00e9rant|g\u00e9rante|pr\u00e9sident|pr\u00e9sidente|directeur|directrice|ceo|dirigeant)\s*[:\-]?\s*([A-Z\u00c0-\u017f'\- ]{2,})",
    re.IGNORECASE,
)
ADDRESS_REGEX = re.compile(
    r"(?:adresse|si\u00e8ge social|siege social|bureau)\s*[:\-]?\s*([^\n]+)",
    re.IGNORECASE,
)
REGION_CODES = ("FR", "BE", "LU", "CH", "DE")
PAGE_PRIORITY_KEYWORDS = (
    (r"mentions", 0),
    (r"legal", 0),
    (r"contact", 1),
    (r"nous-contacter", 1),
    (r"a-propos", 2),
    (r"about", 2),
)


@dataclass
class PageExtraction:
    url: str
    status: int
    priority: int
    emails: Set[str] = field(default_factory=set)
    email_types: Dict[str, str] = field(default_factory=dict)
    phones: Set[str] = field(default_factory=set)
    social: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    sirets: Set[str] = field(default_factory=set)
    sirens: Set[str] = field(default_factory=set)
    rcs_entries: Set[str] = field(default_factory=set)
    legal_managers: Set[str] = field(default_factory=set)
    addresses: Set[str] = field(default_factory=set)
    discovered_at: Optional[str] = None


def _deobfuscate(text: str) -> str:
    result = text
    for pattern, repl in OBFUSCATION_PATTERNS:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    result = re.sub(r'([A-Za-z\u00C0-\u017F]+)\s+([A-Za-z\u00C0-\u017F]+)(?=@)', r'\1.\2', result)
    return result


def _sanitize_email(email: str) -> str:
    email = email.strip().strip(".,;:<>[](){}")
    email = re.sub(r'\s*@\s*', '@', email)
    email = re.sub(r'\s*\.\s*', '.', email)
    if '@' in email:
        local, domain = email.split('@', 1)
        local = re.sub(r'[\s]+', '.', local.strip())
        local = re.sub(r'\.+', '.', local).strip('.')
        domain = re.sub(r'\s+', '', domain)
        email = f'{local}@{domain}'
    return email.lower()


def _extract_emails(text: str) -> Set[str]:
    cleaned = _deobfuscate(text)
    return {_sanitize_email(match) for match in EMAIL_REGEX.findall(cleaned)}


def _classify_email(email: str, generic_prefixes: Set[str]) -> str:
    local, _, _domain = email.partition("@")
    local_lower = local.lower()
    for prefix in generic_prefixes:
        if local_lower.startswith(prefix):
            return "generic"
    if re.match(r"^[a-z]+[._-][a-z]+$", local_lower):
        return "nominative"
    if re.match(r"^[a-z]+\.[a-z]+\.[a-z]+$", local_lower):
        return "nominative"
    if any(char.isdigit() for char in local_lower):
        return "generic"
    if len(local_lower) <= 5:
        return "generic"
    return "nominative"


def _extract_phones(text: str) -> Set[str]:
    raw = _deobfuscate(text)
    phones: Set[str] = set()
    for region in REGION_CODES:
        matcher = phonenumbers.PhoneNumberMatcher(raw, region)
        for match in matcher:
            number = match.number
            if phonenumbers.is_valid_number(number):
                formatted = phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)
                phones.add(formatted)
    return phones


def _extract_social(text: str) -> Dict[str, Set[str]]:
    social: Dict[str, Set[str]] = defaultdict(set)
    for network, pattern in SOCIAL_PATTERNS.items():
        for match in pattern.findall(text):
            cleaned = match.rstrip("/ ")
            social[network].add(cleaned)
    return social


def _extract_sirets(text: str) -> Tuple[Set[str], Set[str]]:
    sirets = {match for match in SIRET_REGEX.findall(text)}
    sirens = {match for match in SIREN_REGEX.findall(text)}
    for siret in sirets:
        sirens.add(siret[:9])
    return sirets, sirens


def _extract_rcs(text: str) -> Set[str]:
    entries = set()
    for match in RCS_REGEX.finditer(text):
        city = (match.group(1) or "").strip()
        number = match.group(2)
        if number:
            if city:
                entries.add(f"RCS {city.title()} {number}")
            else:
                entries.add(f"RCS {number}")
    return entries


def _extract_managers(text: str) -> Set[str]:
    managers = set()
    for match in LEGAL_MANAGER_REGEX.finditer(text):
        name = match.group(2).strip()
        if name and len(name.split()) <= 5:
            managers.add(name.title())
    return managers


def _extract_addresses(text: str, html: str) -> Set[str]:
    addresses = set()
    for match in ADDRESS_REGEX.finditer(text):
        candidate = match.group(1).strip()
        candidate = re.split(r"(?:t\u00e9l\u00e9phone|telephone|email|mail|tel|site)", candidate, flags=re.IGNORECASE)[0]
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if candidate:
            addresses.add(candidate)
    if not addresses and html:
        soup = BeautifulSoup(html, "lxml")
        for label in soup.find_all(string=re.compile(r"adresse", re.IGNORECASE)):
            text = label.parent.get_text(" ", strip=True) if label.parent else str(label)
            if text:
                addresses.add(text)
                break
    return addresses


def _page_priority(url: str) -> int:
    lower_url = url.lower()
    for pattern, score in PAGE_PRIORITY_KEYWORDS:
        if pattern in lower_url:
            return score
    return 5


def run(cfg: dict, ctx: dict) -> dict:
    logger = ctx.get("logger")
    outdir = Path(ctx["outdir"])
    pages_path = outdir / "crawl" / "pages.parquet"
    serp_path = outdir / "serp" / "serp_results.parquet"
    if not pages_path.exists() or not serp_path.exists():
        if logger:
            logger.warning("parse_contacts: missing crawl or serp outputs")
        return {"status": "SKIPPED", "reason": "MISSING_INPUTS"}

    pages_df = pd.read_parquet(pages_path)
    if pages_df.empty:
        return {"status": "SKIPPED", "reason": "EMPTY_PAGES"}

    serp_df = pd.read_parquet(serp_path)
    parse_cfg = (cfg.get("parse") or {})
    prefer_mentions = bool(parse_cfg.get("prefer_mentions_legales"))
    generic_list = set(cfg.get("quality", {}).get("email_generic_list") or [])
    generic_prefixes = {prefix.rstrip("@").lower() for prefix in generic_list}
    generic_prefixes.update(GENERIC_PREFIXES)

    domain_data: Dict[str, List[PageExtraction]] = defaultdict(list)

    for _, row in pages_df.iterrows():
        domain = str(row.get("domain") or "").strip().lower()
        if not domain:
            continue
        url = str(row.get("url") or row.get("requested_url") or "").strip()
        text = str(row.get("content_text") or "")
        html = str(row.get("content_html_trunc") or "")
        status = int(row.get("status") or 0)
        priority = _page_priority(url)
        if prefer_mentions and "mention" in url.lower():
            priority = -1
        extraction = PageExtraction(url=url, status=status, priority=priority)
        extraction.discovered_at = row.get("discovered_at") or None

        if text:
            emails = _extract_emails(text)
            for email in emails:
                extraction.emails.add(email)
                extraction.email_types[email] = _classify_email(email, generic_prefixes)
            extraction.phones = _extract_phones(text)
            social = _extract_social(text)
            for network, urls in social.items():
                extraction.social[network].update(urls)
            sirets, sirens = _extract_sirets(text)
            extraction.sirets = sirets
            extraction.sirens = sirens
            extraction.rcs_entries = _extract_rcs(text)
            extraction.legal_managers = _extract_managers(text)
            extraction.addresses = _extract_addresses(text, html)
        domain_data[domain].append(extraction)

    if not domain_data:
        return {"status": "SKIPPED", "reason": "NO_CONTACTS"}

    aggregated_by_domain: Dict[str, Dict[str, object]] = {}

    for domain, pages in domain_data.items():
        pages.sort(key=lambda p: (p.priority, 0 if p.status < 400 else 1, p.url))
        email_sources: Dict[str, Tuple[int, str, str]] = {}
        for page in pages:
            for email in page.emails:
                meta = (page.priority, page.url, page.email_types.get(email, "generic"))
                existing = email_sources.get(email)
                if existing is None or meta < existing:
                    email_sources[email] = meta
        aggregated_by_domain[domain] = {
            "emails": sorted(email_sources.keys()),
            "email_source_meta": email_sources,
            "phones": sorted({phone for page in pages for phone in page.phones}),
            "social_links": {
                network: sorted({url for page in pages for url in page.social.get(network, set())})
                for network in SOCIAL_PATTERNS.keys()
            },
            "sirets": sorted({siret for page in pages for siret in page.sirets}),
            "sirens": sorted({siren for page in pages for siren in page.sirens}),
            "rcs": sorted({rcs for page in pages for rcs in page.rcs_entries}),
            "legal_managers": sorted({m for page in pages for m in page.legal_managers}),
            "addresses": sorted({addr for page in pages for addr in page.addresses}),
            "best_page": pages[0].url,
            "best_status": pages[0].status,
            "best_discovered_at": pages[0].discovered_at,
        }

    output_rows: List[Dict[str, object]] = []

    for _, row in serp_df.iterrows():
        top_url = str(row.get("top_url") or "").strip()
        domain = str(row.get("top_domain") or "").strip().lower()
        if not domain and top_url:
            domain = registered_domain(top_url)
        if not domain:
            continue
        aggregate = aggregated_by_domain.get(domain, {})
        email_meta = aggregate.get("email_source_meta") or {}
        best_email = None
        best_email_type = None
        if email_meta:
            sorted_emails = sorted(
                email_meta.items(),
                key=lambda kv: (kv[1][0], 0 if kv[1][2] == "nominative" else 1, kv[0]),
            )
            best_email, meta = sorted_emails[0]
            best_email_type = meta[2]
        record = {
            "siren": row.get("siren"),
            "denomination": row.get("denomination"),
            "domain": domain,
            "top_url": top_url,
            "best_page": aggregate.get("best_page"),
            "best_status": aggregate.get("best_status"),
            "best_email": best_email,
            "email_type": best_email_type,
            "emails": aggregate.get("emails") or [],
            "phones": aggregate.get("phones") or [],
            "social_links": aggregate.get("social_links") or {},
            "rcs": aggregate.get("rcs") or [],
            "legal_managers": aggregate.get("legal_managers") or [],
            "addresses": aggregate.get("addresses") or [],
            "sirets": aggregate.get("sirets") or [],
            "sirens_extracted": aggregate.get("sirens") or [],
            "confidence": row.get("confidence"),
            "query": row.get("query"),
            "rank": row.get("rank"),
            "discovered_at": aggregate.get("best_discovered_at"),
        }
        output_rows.append(record)

    contacts_dir = io.ensure_dir(outdir / "contacts")
    parquet_path = contacts_dir / "contacts.parquet"
    jsonl_path = contacts_dir / "contacts.jsonl"

    df = pd.DataFrame(output_rows)
    df.to_parquet(parquet_path, index=False)

    if jsonl_path.exists():
        jsonl_path.unlink()
    for record in output_rows:
        io.log_json(jsonl_path, record)

    return {
        "status": "OK",
        "records": len(output_rows),
        "output": str(parquet_path),
    }
