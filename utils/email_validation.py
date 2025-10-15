"""Email validation helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Final

try:
    import dns.exception
    import dns.resolver
except ImportError:  # pragma: no cover - defensive fallback when dnspython is missing
    dns = None  # type: ignore[assignment]

_MX_TIMEOUT: Final[float] = 2.5


def has_mx_record(domain: str) -> bool:
    """Return ``True`` if the domain exposes at least one MX record."""

    normalized = (domain or "").strip().strip(".").lower()
    if not normalized:
        return False

    if dns is None:  # pragma: no cover - when dnspython is not available
        return False

    return _has_mx_record_normalized(normalized)


@lru_cache(maxsize=2048)
def _has_mx_record_normalized(domain: str) -> bool:
    """Cached MX lookup for already-normalized domains."""

    try:
        answers = dns.resolver.resolve(domain, "MX", lifetime=_MX_TIMEOUT)
        return bool(answers)
    except dns.exception.DNSException:  # pragma: no cover - dns exceptions are not predictable
        return False

__all__ = ["has_mx_record"]
