"""Email validation helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import List
from urllib.parse import urlparse

import dns.resolver

_NAME_TOKEN_RE = re.compile(r"[A-Za-z\u00C0-\u017F0-9']+")
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")
_LOCAL_DISALLOWED_RE = re.compile(r"[^a-z0-9.\-_]")
_MULTI_SEP_RE = re.compile(r"[.\-_]{2,}")


def _strip_accents(value: str) -> str:
    """Return *value* without diacritic marks."""
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _slugify_token(token: str) -> str:
    """Simplify a single name token to lowercase ASCII."""
    if not token:
        return ""
    ascii_token = _strip_accents(token)
    ascii_token = _NON_ALNUM_RE.sub("", ascii_token)
    return ascii_token.lower()


def _clean_local_part(value: str) -> str:
    """Keep email local-part safe characters and trim duplicated separators."""
    if not value:
        return ""
    clean = _LOCAL_DISALLOWED_RE.sub("", value.lower())
    clean = _MULTI_SEP_RE.sub(lambda match: match.group(0)[0], clean)
    clean = clean.lstrip("._-").rstrip("._-")
    return clean


def _normalize_domain(domain: str) -> str:
    """Extract and normalise a domain name from arbitrary input."""
    candidate = (domain or "").strip().lower()
    if not candidate:
        return ""
    candidate = candidate.replace("mailto:", "")
    candidate = candidate.split("@")[-1]
    candidate = candidate.strip()
    if not candidate:
        return ""
    url_like = candidate if "://" in candidate else f"http://{candidate}"
    parsed = urlparse(url_like)
    host = parsed.netloc or parsed.path
    host = host.strip()
    if host.startswith("www."):
        host = host[4:]
    host = host.split("/")[0]
    host = host.split(":")[0]
    host = host.strip().strip(".")
    return host


def _domain_has_mx(cleaned: str) -> bool:
    """Return True if *cleaned* exposes at least one MX record."""
    if not cleaned:
        return False
    try:
        answers = dns.resolver.resolve(cleaned, "MX")
        return len(answers) > 0
    except Exception:
        return False


def has_mx_record(domain: str) -> bool:
    """Return True if the domain exposes at least one MX record."""

    cleaned = _normalize_domain(domain)
    return _domain_has_mx(cleaned)


def generate_email_patterns(full_name: str, domain: str) -> List[str]:
    """
    Generate plausible corporate email addresses for ``full_name`` at ``domain``.

    The function emits a list of common patterns (``prenom.nom@``, ``p.nom@``,
    ``nom.prenom@``...) filtered to domains that expose an MX record.
    """

    domain_clean = _normalize_domain(domain)
    if not domain_clean or not _domain_has_mx(domain_clean):
        return []

    raw_tokens = [part for part in _NAME_TOKEN_RE.findall(full_name or "") if part.strip()]
    slug_tokens = [_slugify_token(token) for token in raw_tokens]
    slug_tokens = [token for token in slug_tokens if token]
    if not slug_tokens:
        return []

    first = slug_tokens[0]
    remaining = slug_tokens[1:] or [first]
    last = remaining[-1]
    last_tokens = remaining
    last_compound = "".join(last_tokens)
    last_dotted = ".".join(last_tokens) if len(last_tokens) > 1 else ""

    if not first or not last:
        return []

    first_initial = first[0]
    last_initial = last[0] if last else ""
    initials = "".join(token[0] for token in slug_tokens if token)

    candidates: List[str] = []
    seen: set[str] = set()

    def add(local_part: str) -> None:
        clean = _clean_local_part(local_part)
        if clean and clean not in seen:
            seen.add(clean)
            candidates.append(f"{clean}@{domain_clean}")

    add(f"{first}.{last}")
    add(f"{first}{last}")
    if last_dotted:
        add(f"{first}.{last_dotted}")
    if last_compound and last_compound != last:
        add(f"{first}{last_compound}")

    add(f"{first_initial}.{last}")
    add(f"{first_initial}{last}")
    if last_compound and last_compound != last:
        add(f"{first_initial}{last_compound}")

    add(f"{last}.{first}")
    add(f"{last}{first}")

    if last_initial:
        add(f"{first}.{last_initial}")
        add(f"{first}{last_initial}")
        add(f"{first_initial}.{last_initial}")
        add(f"{first_initial}{last_initial}")

    add(f"{first}-{last}")
    add(f"{first}_{last}")
    add(f"{last}-{first}")
    add(f"{last}_{first}")

    if last_compound and last_compound != last:
        add(last_compound)

    add(last)
    add(first)

    if initials and len(initials) > 1:
        add(initials)

    return candidates
