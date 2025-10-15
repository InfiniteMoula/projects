"""Email validation helpers."""

from __future__ import annotations

import dns.resolver


def has_mx_record(domain: str) -> bool:
    """Return True if the domain exposes at least one MX record."""

    domain = (domain or "").strip().lower()
    if not domain:
        return False
    try:
        answers = dns.resolver.resolve(domain, "MX")
        return len(answers) > 0
    except Exception:
        return False
