from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib import robotparser
from urllib.parse import urlparse, urlunparse

import httpx

from cache.sqlite_cache import SQLiteCache

LOGGER = logging.getLogger("net.http_client")


@dataclass
class _CacheEntry:
    status: int
    text: str
    headers: Dict[str, str]
    timestamp: float
    user_agent: str


class HttpClient:
    """HTTP client with shared pools, caching, retries, and robots handling."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self._cfg = dict(cfg)
        self._timeout_seconds = float(self._cfg.get("timeout", 8.0))
        self._max_concurrent_requests = max(1, int(self._cfg.get("max_concurrent_requests", 10)))
        self._per_host_limit = max(1, int(self._cfg.get("per_host_limit", 2)))
        self._retry_attempts = max(1, int(self._cfg.get("retry_attempts", 3)))
        self._backoff_factor = float(self._cfg.get("backoff_factor", 0.5))
        self._backoff_max = float(self._cfg.get("max_backoff", 10.0))
        self._respect_robots = bool(self._cfg.get("respect_robots", True))
        self._robots_cache_ttl = float(self._cfg.get("robots_cache_ttl", 3600.0))
        self._default_headers = dict(self._cfg.get("default_headers", {}))
        self._user_agents = self._load_user_agents(self._cfg)

        cache_ttl_days = float(self._cfg.get("cache_ttl_days", 1.0))
        cache_dir_value = self._cfg.get("cache_dir")
        self._cache: Optional[SQLiteCache] = None
        if cache_dir_value and cache_ttl_days > 0:
            cache_dir = Path(cache_dir_value)
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / "http.db"
            self._cache = SQLiteCache(str(db_path), cache_ttl_days)
        self._cache_enabled = self._cache is not None

        limits = httpx.Limits(
            max_connections=int(
                self._cfg.get(
                    "max_connections",
                    max(self._max_concurrent_requests, self._per_host_limit * 4),
                )
            ),
            max_keepalive_connections=int(
                self._cfg.get("max_keepalive_connections", self._max_concurrent_requests)
            ),
        )
        timeout = httpx.Timeout(self._timeout_seconds)
        follow_redirects = bool(self._cfg.get("follow_redirects", True))

        self._sync_client = httpx.Client(timeout=timeout, limits=limits, follow_redirects=follow_redirects)
        self._async_client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
        )

        self._global_semaphore: Optional[asyncio.Semaphore] = None
        self._per_host_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._per_host_lock: Optional[asyncio.Lock] = None

        domain_delays_cfg = self._cfg.get("per_host_delay_seconds", {})
        self._per_host_delays = {
            host.lower(): max(0.0, float(delay)) for host, delay in dict(domain_delays_cfg).items()
        }
        self._default_host_delay = max(0.0, float(self._cfg.get("default_per_host_delay", 0.0)))

        self._async_rate_lock: Optional[asyncio.Lock] = None
        self._async_last_request: Dict[str, float] = {}
        self._sync_rate_lock = threading.Lock()
        self._sync_last_request: Dict[str, float] = {}

        self._robots_cache: Dict[str, Tuple[robotparser.RobotFileParser, float]] = {}
        self._robots_lock = threading.Lock()

    async def fetch_all(self, urls: List[str]) -> Dict[str, Tuple[int, str]]:
        """Fetch URLs concurrently with rate limits and retries."""

        await self._ensure_async_primitives()

        async def _task(url: str) -> Tuple[str, Tuple[int, str]]:
            status, body = await self._fetch_async(url)
            return url, (status, body)

        pairs = await asyncio.gather(*[_task(url) for url in urls])
        return {url: result for url, result in pairs}

    def get(self, url: str) -> httpx.Response:
        """Synchronous GET with retries, caching, and robots controls."""
        ua = self._choose_user_agent()
        try:
            if self._cache_enabled:
                cached = self._read_cache(url)
                if cached:
                    request = httpx.Request("GET", url, headers=self._build_headers(cached.user_agent))
                    response = httpx.Response(
                        status_code=cached.status,
                        headers=cached.headers,
                        content=cached.text.encode("utf-8"),
                        request=request,
                    )
                    response.extensions["from_cache"] = True
                    LOGGER.info("GET %s status=%s duration=0.000 source=cache", url, cached.status)
                    return response

            if not self.allow_fetch(url, ua):
                LOGGER.warning("GET %s blocked by robots", url)
                return self._error_response("GET", url, "blocked_by_robots")

            host = self._extract_host(url)
            self._maybe_wait_sync(host)
            status_response = self._perform_sync_request("GET", url, ua)
            self._mark_sync(host)

            if status_response.status_code == httpx.codes.OK and self._cache_enabled:
                self._write_cache(url, status_response, ua)
            return status_response
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("GET %s unexpected error: %s", url, exc)
            return self._error_response("GET", url, "unexpected_error")

    def head(self, url: str) -> httpx.Response:
        """Synchronous HEAD with retries and robots controls."""
        ua = self._choose_user_agent()
        try:
            if not self.allow_fetch(url, ua):
                LOGGER.warning("HEAD %s blocked by robots", url)
                return self._error_response("HEAD", url, "blocked_by_robots")

            host = self._extract_host(url)
            self._maybe_wait_sync(host)
            response = self._perform_sync_request("HEAD", url, ua)
            self._mark_sync(host)
            return response
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("HEAD %s unexpected error: %s", url, exc)
            return self._error_response("HEAD", url, "unexpected_error")

    def close(self) -> None:
        """Close the underlying httpx clients."""
        self._sync_client.close()

    async def aclose(self) -> None:
        """Close asynchronous resources."""
        await self._async_client.aclose()

    async def _fetch_async(self, url: str) -> Tuple[int, str]:
        await self._ensure_async_primitives()
        ua = self._choose_user_agent()
        try:
            if self._cache_enabled:
                cached = await asyncio.to_thread(self._read_cache, url)
                if cached:
                    LOGGER.info("GET %s status=%s duration=0.000 source=cache", url, cached.status)
                    return cached.status, cached.text

            allowed = await asyncio.to_thread(self.allow_fetch, url, ua)
            if not allowed:
                LOGGER.warning("GET %s blocked by robots", url)
                return 0, ""

            host = self._extract_host(url)
            semaphore = await self._per_host_semaphore(host)
            assert self._global_semaphore is not None  # sanity
            async with self._global_semaphore, semaphore:
                await self._maybe_wait_async(host)
                attempt = 0
                last_text = ""
                last_status = 0
                start = time.perf_counter()
                while attempt < self._retry_attempts:
                    headers = self._build_headers(ua)
                    try:
                        response = await self._async_client.get(url, headers=headers)
                        last_status = response.status_code
                        last_text = response.text
                        duration = time.perf_counter() - start
                        if response.status_code in {429, 503} and attempt + 1 < self._retry_attempts:
                            LOGGER.warning(
                                "GET %s retryable status=%s attempt=%s",
                                url,
                                response.status_code,
                                attempt + 1,
                            )
                            await self._sleep_backoff_async(attempt)
                            attempt += 1
                            continue

                        LOGGER.info("GET %s status=%s duration=%.3f", url, response.status_code, duration)
                        if (
                            response.status_code == httpx.codes.OK
                            and self._cache_enabled
                        ):
                            await asyncio.to_thread(self._write_cache, url, response, ua)
                        break
                    except httpx.TimeoutException as exc:
                        duration = time.perf_counter() - start
                        LOGGER.warning("GET %s timeout attempt=%s duration=%.3f error=%s", url, attempt + 1, duration, exc)
                        last_status = 0
                        last_text = ""
                        if attempt + 1 >= self._retry_attempts:
                            break
                        await self._sleep_backoff_async(attempt)
                    except httpx.HTTPError as exc:
                        duration = time.perf_counter() - start
                        LOGGER.warning("GET %s http error attempt=%s duration=%.3f error=%s", url, attempt + 1, duration, exc)
                        last_status = 0
                        last_text = ""
                        if attempt + 1 >= self._retry_attempts:
                            break
                        await self._sleep_backoff_async(attempt)
                    attempt += 1
                await self._mark_async(host)
                return last_status, last_text
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("GET %s unexpected error: %s", url, exc)
            return 0, ""

    def _perform_sync_request(self, method: str, url: str, ua: str) -> httpx.Response:
        headers = self._build_headers(ua)
        attempt = 0
        last_response: Optional[httpx.Response] = None
        start = time.perf_counter()
        while attempt < self._retry_attempts:
            try:
                response = self._sync_client.request(method, url, headers=headers)
                last_response = response
                duration = time.perf_counter() - start
                if response.status_code in {429, 503} and attempt + 1 < self._retry_attempts:
                    LOGGER.warning(
                        "%s %s retryable status=%s attempt=%s",
                        method,
                        url,
                        response.status_code,
                        attempt + 1,
                    )
                    self._sleep_backoff_sync(attempt)
                    attempt += 1
                    continue
                LOGGER.info("%s %s status=%s duration=%.3f", method, url, response.status_code, duration)
                return response
            except httpx.TimeoutException as exc:
                duration = time.perf_counter() - start
                LOGGER.warning("%s %s timeout attempt=%s duration=%.3f error=%s", method, url, attempt + 1, duration, exc)
                if attempt + 1 >= self._retry_attempts:
                    break
                self._sleep_backoff_sync(attempt)
            except httpx.HTTPError as exc:
                duration = time.perf_counter() - start
                LOGGER.warning("%s %s http error attempt=%s duration=%.3f error=%s", method, url, attempt + 1, duration, exc)
                if attempt + 1 >= self._retry_attempts:
                    break
                self._sleep_backoff_sync(attempt)
            attempt += 1
        if last_response is not None:
            return last_response
        return self._error_response(method, url, "request_failed")

    def allow_fetch(self, url: str, ua: str) -> bool:
        """Return False when robots.txt forbids fetching the given URL."""
        if not self._respect_robots:
            return True

        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host:
            return True
        path = parsed.path or "/"
        parser = self._get_robot_parser(parsed, ua, host)
        try:
            return parser.can_fetch(ua, path)
        except Exception:  # pragma: no cover - defensive
            return True

    def _get_robot_parser(
        self,
        parsed_url,
        ua: str,
        host: str,
    ) -> robotparser.RobotFileParser:
        now = time.monotonic()
        with self._robots_lock:
            entry = self._robots_cache.get(host)
            if entry and now - entry[1] < self._robots_cache_ttl:
                return entry[0]

        parser = robotparser.RobotFileParser()
        scheme = parsed_url.scheme or "https"
        netloc = parsed_url.netloc or parsed_url.hostname or host
        robots_url = urlunparse((scheme, netloc, "/robots.txt", "", "", ""))
        parser.set_url(robots_url)
        try:
            response = self._sync_client.get(
                robots_url,
                headers={
                    "User-Agent": ua,
                    "Accept": "text/plain, */*;q=0.5",
                },
            )
        except httpx.HTTPError as exc:
            LOGGER.debug("robots fetch failed for %s: %s", robots_url, exc)
            parser.parse(["User-agent: *", "Disallow:"])
            with self._robots_lock:
                self._robots_cache[host] = (parser, now)
            return parser

        status = response.status_code
        if status == 404:
            parser.parse(["User-agent: *", "Disallow:"])
        elif status in {401, 403}:
            parser.parse(["User-agent: *", "Disallow: /"])
        elif status >= 400:
            LOGGER.debug("robots fetch unexpected status %s for %s", status, robots_url)
            parser.parse(["User-agent: *", "Disallow:"])
        else:
            content = response.text or ""
            parser.parse(content.splitlines())

        with self._robots_lock:
            self._robots_cache[host] = (parser, now)
        return parser

    def _load_user_agents(self, cfg: Mapping[str, Any]) -> List[str]:
        ua_list: List[str] = []
        maybe_list: Optional[Iterable[str]] = cfg.get("user_agents")
        if maybe_list:
            ua_list.extend([ua.strip() for ua in maybe_list if isinstance(ua, str) and ua.strip()])
        ua_file = cfg.get("user_agents_file")
        if ua_file:
            try:
                path = Path(ua_file)
                if path.exists():
                    lines = path.read_text(encoding="utf-8").splitlines()
                    ua_list.extend([line.strip() for line in lines if line.strip()])
            except OSError as exc:
                LOGGER.warning("Unable to load user agents from %s: %s", ua_file, exc)
        if not ua_list:
            ua_list.append("Mozilla/5.0 (compatible; HttpClient/1.0; +https://example.com/bot)")
        return ua_list

    def _choose_user_agent(self) -> str:
        return random.choice(self._user_agents)

    def _build_headers(self, ua: str) -> Dict[str, str]:
        headers = dict(self._default_headers)
        headers.setdefault("User-Agent", ua)
        headers.setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        headers.setdefault("Accept-Language", "en-US,en;q=0.5")
        return headers

    def _read_cache(self, url: str) -> Optional[_CacheEntry]:
        if not self._cache_enabled or self._cache is None:
            return None
        payload = self._cache.get(url)
        if not isinstance(payload, dict):
            return None
        headers_map: Dict[str, str] = {}
        raw_headers = payload.get("headers", {})
        if isinstance(raw_headers, dict):
            headers_map = {str(k): str(v) for k, v in raw_headers.items()}
        elif isinstance(raw_headers, list):
            headers_map = {str(k): str(v) for k, v in raw_headers}
        try:
            status = int(payload.get("status", 0))
            text = str(payload.get("text", ""))
            timestamp = float(payload.get("timestamp", time.time()))
            user_agent = str(payload.get("user_agent", self._choose_user_agent()))
        except (TypeError, ValueError):
            return None
        return _CacheEntry(
            status=status,
            text=text,
            headers=headers_map,
            timestamp=timestamp,
            user_agent=user_agent,
        )

    def _write_cache(self, url: str, response: httpx.Response, ua: str) -> None:
        if not self._cache_enabled or self._cache is None:
            return
        record = {
            "status": response.status_code,
            "text": response.text,
            "headers": dict(response.headers.items()),
            "timestamp": time.time(),
            "user_agent": ua,
        }
        self._cache.set(url, record)

    async def _per_host_semaphore(self, host: str) -> asyncio.Semaphore:
        await self._ensure_async_primitives()
        key = host or ""
        assert self._per_host_lock is not None
        async with self._per_host_lock:
            semaphore = self._per_host_semaphores.get(key)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self._per_host_limit)
                self._per_host_semaphores[key] = semaphore
            return semaphore

    async def _maybe_wait_async(self, host: str) -> None:
        await self._ensure_async_primitives()
        delay = self._per_host_delays.get(host, self._default_host_delay)
        if delay <= 0:
            return
        assert self._async_rate_lock is not None
        async with self._async_rate_lock:
            last = self._async_last_request.get(host)
            if last is None:
                return
            wait_time = delay - (time.monotonic() - last)
        if wait_time > 0:
            await asyncio.sleep(wait_time)

    async def _mark_async(self, host: str) -> None:
        if not host and host != "":
            return
        await self._ensure_async_primitives()
        assert self._async_rate_lock is not None
        async with self._async_rate_lock:
            self._async_last_request[host] = time.monotonic()

    def _maybe_wait_sync(self, host: str) -> None:
        delay = self._per_host_delays.get(host, self._default_host_delay)
        if delay <= 0:
            return
        with self._sync_rate_lock:
            last = self._sync_last_request.get(host)
            if last is None:
                return
            wait_time = delay - (time.monotonic() - last)
        if wait_time > 0:
            time.sleep(wait_time)

    def _mark_sync(self, host: str) -> None:
        if not host and host != "":
            return
        with self._sync_rate_lock:
            self._sync_last_request[host] = time.monotonic()

    async def _sleep_backoff_async(self, attempt: int) -> None:
        delay = min(self._backoff_factor * (2 ** attempt), self._backoff_max)
        jitter = random.uniform(0.5, 1.5)
        await asyncio.sleep(delay * jitter)

    async def _ensure_async_primitives(self) -> None:
        if self._global_semaphore is None:
            self._global_semaphore = asyncio.Semaphore(self._max_concurrent_requests)
        if self._per_host_lock is None:
            self._per_host_lock = asyncio.Lock()
        if self._async_rate_lock is None:
            self._async_rate_lock = asyncio.Lock()

    def _sleep_backoff_sync(self, attempt: int) -> None:
        delay = min(self._backoff_factor * (2 ** attempt), self._backoff_max)
        jitter = random.uniform(0.5, 1.5)
        time.sleep(delay * jitter)

    def _extract_host(self, url: str) -> str:
        parsed = urlparse(url)
        return (parsed.hostname or "").lower()

    def _error_response(self, method: str, url: str, reason: str) -> httpx.Response:
        request = httpx.Request(method, url)
        content = reason.encode("utf-8")
        return httpx.Response(status_code=0, request=request, content=content, reason=reason)
