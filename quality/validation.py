"""Validation helpers for quality checks.

Each helper returns a :class:`FieldValidation` describing whether the supplied
value is acceptable together with optional normalization hints.  The rules
capture the business expectations for public contact details:

- Websites must be HTTP(S) URLs pointing to a registrable domain.
- Emails must follow the RFC-style format and expose a reachable MX record.
- Telephones must be valid French numbers expressed in E.164 form.
- LinkedIn URLs must target company pages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

import dns.exception
import dns.resolver
import phonenumbers
import tldextract
from email.utils import parseaddr


@dataclass(frozen=True)
class FieldValidation:
    """Container describing the outcome of a field validation."""

    is_valid: bool
    normalized: Optional[str] = None
    reason: Optional[str] = None
    details: Optional[Dict[str, object]] = None


_EMAIL_RE = re.compile(
    r"^(?:[-!#$%&'*+/0-9=?A-Z^_`a-z{|}~]+(\.[-!#$%&'*+/0-9=?A-Z^_`a-z{|}~]+)*)@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}$"
)
_LINKEDIN_COMPANY_RE = re.compile(
    r"^/company/([A-Za-z0-9-]{2,})/?([A-Za-z0-9\-/]*)?$", re.IGNORECASE
)


def _normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def validate_site_web(value) -> FieldValidation:
    """Validate that the URL starts with HTTP(S) and targets a registrable domain."""
    text = _normalize_text(value)
    if not text:
        return FieldValidation(False, reason="missing")

    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return FieldValidation(False, normalized=text, reason="scheme")
    if not parsed.netloc:
        return FieldValidation(False, normalized=text, reason="netloc")

    netloc = parsed.netloc.lower()
    extract = tldextract.extract(netloc)
    if not extract.domain or not extract.suffix:
        return FieldValidation(False, normalized=text, reason="domain")

    normalized = f"{scheme}://{netloc}{parsed.path or ''}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    if parsed.fragment:
        normalized += f"#{parsed.fragment}"
    return FieldValidation(True, normalized=normalized)


def validate_email(
    value,
    *,
    check_mx: bool = True,
    mx_cache: Optional[Dict[str, Optional[bool]]] = None,
) -> FieldValidation:
    """Validate email format (RFC-like) and optionally ensure MX record exists."""
    text = _normalize_text(value)
    if not text:
        return FieldValidation(False, reason="missing")

    _, address = parseaddr(text)
    if not address:
        return FieldValidation(False, normalized=text, reason="parse")

    if not _EMAIL_RE.fullmatch(address):
        return FieldValidation(False, normalized=address, reason="format")

    normalized = address.strip()
    domain = normalized.rsplit("@", 1)[1].lower()
    local_part = normalized.rsplit("@", 1)[0]
    normalized = f"{local_part}@{domain}"
    details: Dict[str, object] = {"mx_checked": bool(check_mx)}
    if check_mx:
        try:
            if mx_cache is not None and domain in mx_cache:
                mx_valid = mx_cache[domain]
                details["mx_valid"] = mx_valid
                if mx_valid is False:
                    return FieldValidation(False, normalized=normalized, reason="mx", details=details)
                return FieldValidation(True, normalized=normalized, details=details)
            answers = dns.resolver.resolve(domain, "MX")
            records = [rdata for rdata in answers]
            has_records = bool(records)
            details["mx_valid"] = has_records
            if mx_cache is not None:
                mx_cache[domain] = has_records
            if not has_records:
                return FieldValidation(False, normalized=normalized, reason="mx", details=details)
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
            details["mx_valid"] = False
            if mx_cache is not None:
                mx_cache[domain] = False
            return FieldValidation(False, normalized=normalized, reason="mx", details=details)
        except (dns.exception.DNSException, TimeoutError):
            # Network/lookup issues should not fail formatting validation outright.
            details["mx_valid"] = None
            details["mx_error"] = "lookup_failed"
            if mx_cache is not None:
                mx_cache[domain] = None
    return FieldValidation(True, normalized=normalized, details=details)


def validate_telephone(value) -> FieldValidation:
    """Validate that the phone number is a French E.164 formatted number."""
    text = _normalize_text(value)
    if not text:
        return FieldValidation(False, reason="missing")

    try:
        parsed = phonenumbers.parse(text, region="FR")
    except phonenumbers.NumberParseException:
        return FieldValidation(False, normalized=text, reason="parse")

    if not phonenumbers.is_possible_number(parsed):
        return FieldValidation(False, normalized=text, reason="possible")
    if not phonenumbers.is_valid_number(parsed):
        return FieldValidation(False, normalized=text, reason="valid")

    region = phonenumbers.region_code_for_number(parsed)
    if region != "FR":
        return FieldValidation(False, normalized=text, reason="region")

    normalized = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    if not normalized.startswith("+"):
        return FieldValidation(False, normalized=normalized, reason="format")
    return FieldValidation(True, normalized=normalized)


def validate_linkedin_url(value) -> FieldValidation:
    """Validate LinkedIn company URLs (https://www.linkedin.com/company/...)."""
    text = _normalize_text(value)
    if not text:
        return FieldValidation(False, reason="missing")

    parsed = urlparse(text)
    if parsed.scheme != "https":
        return FieldValidation(False, normalized=text, reason="scheme")
    host = parsed.netloc.lower()
    if host not in {"www.linkedin.com", "linkedin.com"}:
        return FieldValidation(False, normalized=text, reason="host")

    match = _LINKEDIN_COMPANY_RE.fullmatch(parsed.path)
    if not match:
        return FieldValidation(False, normalized=text, reason="path")

    slug = match.group(1).lower()
    normalized = f"https://www.linkedin.com/company/{slug}/"
    return FieldValidation(True, normalized=normalized)


__all__ = [
    "FieldValidation",
    "validate_email",
    "validate_linkedin_url",
    "validate_site_web",
    "validate_telephone",
]
