"""Lightweight robots.txt cache.""" 
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, urlunparse
from urllib import robotparser

import httpx

LOGGER = logging.getLogger("utils.robots")


@dataclass
class _RobotsEntry:
    parser: robotparser.RobotFileParser
    fetched_at: float
    status: int


class RobotsCache:
    """Fetch and cache robots.txt per host."""

    def __init__(
        self,
        user_agent: str,
        *,
        timeout: float = 5.0,
        cache_ttl: float = 3600.0,
        max_entries: int = 512,
    ) -> None:
        self._user_agent = user_agent
        self._timeout = timeout
        self._ttl = cache_ttl
        self._max_entries = max_entries
        self._entries: "OrderedDict[str, _RobotsEntry]" = OrderedDict()
        self._lock = threading.Lock()

    def allowed(self, url: str, respect_robots: bool = True) -> bool:
        """Return True if *url* is crawlable for the configured user agent."""

        if not respect_robots or not url:
            return True
        parsed = urlparse(url)
        if not parsed.hostname:
            return True
        parser = self._get_parser(parsed)
        if parser is None:
            return True
        path = parsed.path or "/"
        return parser.can_fetch(self._user_agent, path)

    def crawl_delay(self, url: str) -> Optional[float]:
        """Return crawl-delay directive if defined for the user agent."""

        parsed = urlparse(url)
        if not parsed.hostname:
            return None
        parser = self._get_parser(parsed)
        if parser is None:
            return None
        delay = parser.crawl_delay(self._user_agent)
        return float(delay) if delay is not None else None

    def _get_parser(self, parsed_url) -> Optional[robotparser.RobotFileParser]:
        host = (parsed_url.hostname or "").lower()
        if not host:
            return None
        key = host
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry and now - entry.fetched_at < self._ttl:
                self._entries.move_to_end(key)
                return entry.parser
        robots_entry = self._fetch_robots(parsed_url)
        if robots_entry is None:
            return None
        with self._lock:
            self._entries[key] = robots_entry
            self._entries.move_to_end(key)
            if len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return robots_entry.parser

    def _fetch_robots(self, parsed_url) -> Optional[_RobotsEntry]:
        scheme = parsed_url.scheme or "https"
        netloc = parsed_url.netloc or parsed_url.hostname
        robots_url = urlunparse((scheme, netloc, "/robots.txt", "", "", ""))
        parser = robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                response = client.get(
                    robots_url,
                    headers={
                        "User-Agent": self._user_agent,
                        "Accept": "text/plain, */*;q=0.5",
                    },
                )
        except httpx.HTTPError as exc:
            LOGGER.debug("robots fetch failed for %s: %s", robots_url, exc)
            return _RobotsEntry(parser=self._allow_all(parser), fetched_at=time.monotonic(), status=0)

        status = response.status_code
        if status == 404:
            return _RobotsEntry(parser=self._allow_all(parser), fetched_at=time.monotonic(), status=status)
        if status in {401, 403}:
            return _RobotsEntry(parser=self._disallow_all(parser), fetched_at=time.monotonic(), status=status)
        if status >= 400:
            LOGGER.debug("robots fetch unexpected status %s for %s", status, robots_url)
            return _RobotsEntry(parser=self._allow_all(parser), fetched_at=time.monotonic(), status=status)

        content = response.text or ""
        parser.parse(content.splitlines())
        return _RobotsEntry(parser=parser, fetched_at=time.monotonic(), status=status)

    @staticmethod
    def _allow_all(parser: robotparser.RobotFileParser) -> robotparser.RobotFileParser:
        parser.parse(["User-agent: *", "Disallow:"])
        return parser

    @staticmethod
    def _disallow_all(parser: robotparser.RobotFileParser) -> robotparser.RobotFileParser:
        parser.parse(["User-agent: *", "Disallow: /"])
        return parser
