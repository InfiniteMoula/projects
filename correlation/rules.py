"""Pure correlation rules used by the coherence scorer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional
from urllib.parse import urlparse

import phonenumbers
import tldextract

from constants import GENERIC_EMAIL_DOMAINS, GENERIC_EMAIL_PREFIXES


@dataclass(frozen=True)
class RuleResult:
    """Represent the contribution of a rule to the coherence score."""

    delta: int
    flag: Optional[str] = None


_GENERIC_EMAIL_DOMAINS = {item.lower() for item in GENERIC_EMAIL_DOMAINS if item}
_GENERIC_EMAIL_PREFIXES = {item.lower() for item in GENERIC_EMAIL_PREFIXES if item}

_COUNTRY_NAME_OVERRIDES = {
    "france": "FR",
    "belgique": "BE",
    "belgium": "BE",
    "espagne": "ES",
    "spain": "ES",
    "suisse": "CH",
    "switzerland": "CH",
    "allemagne": "DE",
    "germany": "DE",
    "italie": "IT",
    "italy": "IT",
}


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_registrable_domain(value: str) -> Optional[str]:
    text = _normalize_text(value).lower()
    if not text:
        return None
    if "@" in text:
        text = text.rsplit("@", 1)[1]
    parsed = urlparse(text if text.startswith("http") else f"//{text}", scheme="http")
    host = parsed.netloc or parsed.path
    if not host:
        return None
    extract = tldextract.extract(host)
    if not extract.domain or not extract.suffix:
        return None
    return f"{extract.domain}.{extract.suffix}".lower()


def normalize_country(value: object, default: Optional[str] = "FR", overrides: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Return an ISO alpha-2 country code when possible."""

    text = _normalize_text(value)
    if not text and default:
        text = str(default)
    if not text:
        return None
    text_upper = text.upper()
    if len(text_upper) == 2 and text_upper.isalpha():
        return text_upper
    text_lower = text.lower()
    mapping = {**_COUNTRY_NAME_OVERRIDES}
    if overrides:
        mapping.update({str(k).strip().lower(): str(v).strip().upper() for k, v in overrides.items() if str(v).strip()})
    return mapping.get(text_lower, text_upper if len(text_upper) == 2 else (str(default).upper() if default else None))


def email_domain_alignment(email: object, site_url: object) -> Optional[RuleResult]:
    """Check whether the email domain matches the website domain."""

    email_text = _normalize_text(email).lower()
    site_domain = _extract_registrable_domain(_normalize_text(site_url))
    if not email_text or "@" not in email_text or not site_domain:
        return None
    email_domain = _extract_registrable_domain(email_text)
    if not email_domain:
        return None
    if email_domain == site_domain or email_domain.endswith(f".{site_domain}"):
        return RuleResult(30, "email_domain_match")
    return RuleResult(-20, "email_domain_mismatch")


def generic_email_penalty(email: object) -> Optional[RuleResult]:
    """Penalise generic mailbox patterns."""

    email_text = _normalize_text(email).lower()
    if not email_text or "@" not in email_text:
        return None
    local_part, domain_part = email_text.rsplit("@", 1)
    registrable = _extract_registrable_domain(domain_part)
    if registrable and registrable in _GENERIC_EMAIL_DOMAINS:
        return RuleResult(-10, "generic_email_domain")
    if not local_part:
        return None
    simplified = local_part.replace(".", " ").replace("_", " ").split()
    for token in simplified or [local_part]:
        if token.lower() in _GENERIC_EMAIL_PREFIXES:
            return RuleResult(-10, "generic_email_localpart")
    return None


def phone_country_alignment(phone: object, expected_country: Optional[str]) -> Optional[RuleResult]:
    """Verify that the phone number matches the expected country code."""

    phone_text = _normalize_text(phone)
    if not phone_text or not expected_country:
        return None
    try:
        parsed = phonenumbers.parse(phone_text, expected_country)
    except phonenumbers.NumberParseException:
        return RuleResult(-15, "phone_unparseable")
    region = phonenumbers.region_code_for_number(parsed)
    if region == expected_country:
        return RuleResult(20, "phone_country_match")
    return RuleResult(-20, "phone_country_mismatch")


def linkedin_site_alignment(linkedin_url: object, site_url: object) -> Optional[RuleResult]:
    """Check whether the LinkedIn slug aligns with the website domain."""

    linkedin_text = _normalize_text(linkedin_url)
    site_domain = _extract_registrable_domain(_normalize_text(site_url))
    if not linkedin_text or not site_domain:
        return None
    parsed = urlparse(linkedin_text)
    path_parts = [part for part in parsed.path.lower().split("/") if part]
    if not path_parts:
        return None
    if path_parts[0] not in {"company", "in"} or len(path_parts) < 2:
        return RuleResult(-15, "linkedin_path_unexpected")
    slug = path_parts[1]
    domain_token = site_domain.split(".")[0].replace("-", "").lower()
    raw_tokens = (
        slug.replace("%20", " ")
        .replace("-", " ")
        .replace("_", " ")
        .split()
    )
    if not raw_tokens:
        raw_tokens = [slug.replace("-", "").replace("_", "").lower()]
    for token in raw_tokens:
        cleaned = token.lower()
        if not cleaned:
            continue
        if domain_token and (
            cleaned == domain_token
            or cleaned.startswith(domain_token)
            or domain_token.startswith(cleaned)
        ):
            return RuleResult(20, "linkedin_site_match")
    return RuleResult(-15, "linkedin_site_mismatch")


__all__ = [
    "RuleResult",
    "email_domain_alignment",
    "generic_email_penalty",
    "linkedin_site_alignment",
    "normalize_country",
    "phone_country_alignment",
]
