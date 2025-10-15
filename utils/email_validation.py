"""Email validation helpers."""

from __future__ import annotations

try:
    import dns.resolver
except ImportError:  # pragma: no cover - defensive fallback
    dns = None  # type: ignore


def has_mx_record(domain: str) -> bool:
    """Return ``True`` if the domain exposes at least one MX record."""

    if not domain:
        return False

    if dns is None:  # pragma: no cover - when dnspython is not available
        return False

    try:
        answers = dns.resolver.resolve(domain, "MX")
        return len(answers) > 0
    except Exception:  # pragma: no cover - dns exceptions are not predictable
        return False


__all__ = ["has_mx_record"]
