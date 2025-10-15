from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional, Tuple
from urllib import robotparser
from urllib.parse import urlparse, urlunparse

from net.http_client import HttpClient

LOGGER = logging.getLogger("net.robots")

_ROBOTS_CACHE: Dict[str, Tuple[robotparser.RobotFileParser, float]] = {}
_CACHE_LOCK = threading.Lock()
_HTTP_CLIENT: Optional[HttpClient] = None
_CACHE_TTL = 3600.0


def configure(http_client: HttpClient, *, cache_ttl: Optional[float] = None) -> None:
    """Configure the shared HttpClient and optional cache TTL for robots queries."""
    global _HTTP_CLIENT, _CACHE_TTL
    _HTTP_CLIENT = http_client
    if cache_ttl is not None:
        try:
            _CACHE_TTL = max(0.0, float(cache_ttl))
        except (TypeError, ValueError):
            LOGGER.debug("Ignoring invalid robots cache TTL: %r", cache_ttl)


def clear_cache() -> None:
    """Reset the in-memory robots cache. Intended for tests."""
    with _CACHE_LOCK:
        _ROBOTS_CACHE.clear()


def is_allowed(url: str, user_agent: str) -> bool:
    """Return ``True`` when *url* is allowed to be fetched for *user_agent*."""
    if not url:
        return True
    parser = get_parser(url, user_agent)
    path = urlparse(url).path or "/"
    try:
        return parser.can_fetch(user_agent, path)
    except Exception:
        return True


def get_parser(url: str, user_agent: str) -> robotparser.RobotFileParser:
    """
    Return a ``RobotFileParser`` instance for *url*, using an in-memory cache.
    The parser is refreshed according to the configured cache TTL.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        parser = robotparser.RobotFileParser()
        parser.parse(["User-agent: *", "Disallow:"])
        return parser

    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _ROBOTS_CACHE.get(host)
        if cached and (now - cached[1] < _CACHE_TTL):
            return cached[0]

    parser = robotparser.RobotFileParser()
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or parsed.hostname or host
    robots_url = urlunparse((scheme, netloc, "/robots.txt", "", "", ""))
    parser.set_url(robots_url)

    client = _ensure_http_client()
    try:
        response = client.get(robots_url)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("robots fetch failed for %s: %s", robots_url, exc)
        parser.parse(["User-agent: *", "Disallow:"])
        _store(host, parser, now)
        return parser

    status = response.status_code
    text = response.text or ""
    if status == 404:
        parser.parse(["User-agent: *", "Disallow:"])
    elif status in {401, 403}:
        parser.parse(["User-agent: *", "Disallow: /"])
    elif status >= 400 or not text:
        LOGGER.debug("robots unexpected status %s for %s", status, robots_url)
        parser.parse(["User-agent: *", "Disallow:"])
    else:
        parser.parse(text.splitlines())

    _store(host, parser, now)
    return parser


def _store(host: str, parser: robotparser.RobotFileParser, timestamp: float) -> None:
    with _CACHE_LOCK:
        _ROBOTS_CACHE[host] = (parser, timestamp)


def _ensure_http_client() -> HttpClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        LOGGER.debug("Initializing default HttpClient for robots module")
        _HTTP_CLIENT = HttpClient({"timeout": 5.0, "respect_robots": False})
    return _HTTP_CLIENT


__all__ = ["configure", "clear_cache", "get_parser", "is_allowed"]
