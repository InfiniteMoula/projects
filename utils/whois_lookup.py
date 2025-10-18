"""Utilities for extracting WHOIS contact information."""
from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

try:
    import whois  # type: ignore
except ImportError:  # pragma: no cover - dependency should exist at runtime
    whois = None  # type: ignore[assignment]

_EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PRIVACY_KEYWORDS = {
    "redacted",
    "privacy",
    "whoisguard",
    "protected",
    "anonymised",
    "anonymized",
    "not.disclosed",
    "nondisclosed",
    "domain@customer",
    "contact.gandi",
    "mask",
    "data.protected",
    "data-protected",
    "gdpr",
    "please contact",
    "domainadmin@",
    "proxy",
}

_LOGGER = logging.getLogger(__name__)


def _iter_emails(value: object) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _EMAIL_PATTERN.findall(value)
    if isinstance(value, dict):
        emails: list[str] = []
        for nested in value.values():
            emails.extend(_iter_emails(nested))
        return emails
    if isinstance(value, (set, tuple, list)):
        emails: list[str] = []
        for item in value:
            emails.extend(_iter_emails(item))
        return emails
    return []


def _is_public_email(email: str) -> bool:
    normalized = email.strip().lower()
    if not normalized:
        return False
    if any(keyword in normalized for keyword in _PRIVACY_KEYWORDS):
        return False
    return True


def get_public_whois_email(domain: str) -> Optional[str]:
    """Return the first public registrant/admin email address for *domain*.

    The function falls back gracefully when WHOIS information is hidden behind
    a privacy service or the lookup fails.
    """

    if whois is None:
        raise RuntimeError("python-whois is required to query WHOIS data")

    try:
        record = whois.whois(domain)
    except Exception as exc:  # pragma: no cover - depends on network/registry behaviour
        _LOGGER.debug("WHOIS lookup failed for %s: %s", domain, exc)
        return None

    if not record:
        return None

    candidate_keys = [
        "emails",
        "email",
        "registrant_email",
        "registrant_contact_email",
        "admin_email",
        "administrative_contact_email",
        "technical_contact_email",
    ]

    for key in candidate_keys:
        value = getattr(record, key, None)
        if value is None and isinstance(record, dict):
            value = record.get(key)
        if not value:
            continue
        for email in _iter_emails(value):
            if _is_public_email(email):
                return email.lower()

    # Fallback: scan entire record for any email-like string.
    if isinstance(record, dict):
        search_space = record.values()
    else:
        try:
            search_space = vars(record).values()
        except TypeError:
            search_space = []

    for value in search_space:
        for email in _iter_emails(value):
            if _is_public_email(email):
                return email.lower()

    return None
