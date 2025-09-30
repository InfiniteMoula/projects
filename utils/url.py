"""URL and domain helpers."""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse, urlunparse

import tldextract

_WHITESPACE_RE = re.compile(r"\s+")


def ensure_url(value: str, default_scheme: str = "https") -> str:
    """Return *value* with a scheme if missing and trimmed of whitespace."""

    if not value:
        return value
    trimmed = _WHITESPACE_RE.sub(" ", value).strip()
    if "//" not in trimmed:
        return f"{default_scheme}://{trimmed}"
    # Allow schemeless URLs like //example.com
    parsed = urlparse(trimmed, scheme=default_scheme)
    if not parsed.scheme:
        parsed = parsed._replace(scheme=default_scheme)
    return urlunparse(parsed)


def canonicalize(url: str, default_scheme: str = "https", keep_query: bool = False) -> str:
    """Return a normalised URL (lower-cased scheme/host, no fragment)."""

    if not url:
        return url
    parsed = urlparse(url, scheme=default_scheme)
    scheme = parsed.scheme.lower() if parsed.scheme else default_scheme
    host = (parsed.hostname or "").lower()
    port = parsed.port
    netloc = host
    if port and port not in {80, 443}:
        netloc = f"{host}:{port}"
    elif parsed.netloc and not host:
        netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if not keep_query:
        query = ""
    else:
        query = parsed.query
    cleaned = parsed._replace(
        scheme=scheme,
        netloc=netloc,
        path=path if path.startswith("/") else f"/{path}",
        params="",
        query=query,
        fragment="",
    )
    return urlunparse(cleaned)


def resolve(base_url: str, link: str) -> str:
    """Resolve *link* relative to *base_url* and canonicalize it."""

    if not link:
        return ""
    joined = urljoin(base_url, link)
    return canonicalize(joined)


def hostname(url: str) -> str:
    """Return the hostname (lower-cased) for *url*."""

    if not url:
        return ""
    parsed = urlparse(url)
    return (parsed.hostname or "").lower()


def registered_domain(value: str) -> str:
    """Return the registered domain for *value* (domain + public suffix)."""

    if not value:
        return ""
    extracted = tldextract.extract(value)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()
    if extracted.domain:
        return extracted.domain.lower()
    if extracted.suffix:
        return extracted.suffix.lower()
    return value.lower()


def same_registered_domain(left: str, right: str) -> bool:
    """Return True if *left* and *right* share the same registered domain."""

    if not left or not right:
        return False
    return registered_domain(left) == registered_domain(right)


def strip_fragment(url: str) -> str:
    """Remove fragments from *url*."""

    if not url:
        return url
    parsed = urlparse(url)
    return urlunparse(parsed._replace(fragment=""))


def looks_like_home(url: str) -> bool:
    """Heuristic to detect home pages (used to prioritise BFS)."""

    if not url:
        return False
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return path in ("", "/", "/index", "/index.html")
