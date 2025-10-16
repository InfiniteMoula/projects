from __future__ import annotations

import asyncio
import random
import sqlite3
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib import robotparser
from urllib.parse import urlparse, urlunparse

import httpx

from cache.sqlite_cache import CachedResponse, SqliteCache
from metrics.collector import get_metrics
from utils.loggingx import get_logger

LOGGER = get_logger("net.http_client")
METRICS = get_metrics()

class RequestLimiter:
    """Shared limiter coordinating synchronous and asynchronous HTTP request concurrency."""

    def __init__(self, limit: int) -> None:
        self._limit = max(1, int(limit))
        self._semaphore = threading.Semaphore(self._limit)

    @contextmanager
    def sync(self) -> Iterable[None]:
        self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()

    @asynccontextmanager
    async def async_acquire(self) -> Iterable[None]:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._semaphore.acquire)
        try:
            yield
        finally:
            self._semaphore.release()

    @property
    def limit(self) -> int:
        return self._limit


class HttpClient:
    """HTTP client with shared pools, caching, retries, and robots handling."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        cfg_dict = dict(cfg)
        limiter_candidate = cfg_dict.pop("shared_request_limiter", None)
        self._shared_limiter: Optional[RequestLimiter]
        if isinstance(limiter_candidate, RequestLimiter):
            self._shared_limiter = limiter_candidate
        else:
            self._shared_limiter = None
        self._cfg = cfg_dict
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

        self._cache: Optional[SqliteCache] = None
        self._cache_hits = 0
        self._cache_lookups = 0
        self._cache_stats_lock = threading.Lock()
        cache_cfg = self._cfg.get("cache")
        if isinstance(cache_cfg, Mapping):
            enabled = bool(cache_cfg.get("enabled", False))
            if enabled:
                db_path = cache_cfg.get("db_path")
                ttl_raw = cache_cfg.get("ttl_seconds", 0)
                max_raw = cache_cfg.get("max_items", 0)
                try:
                    ttl_seconds = int(ttl_raw)
                except (TypeError, ValueError):
                    ttl_seconds = 0
                try:
                    max_items = int(max_raw)
                except (TypeError, ValueError):
                    max_items = 0
                if db_path and ttl_seconds > 0:
                    cache_path = Path(str(db_path))
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        self._cache = SqliteCache(str(cache_path), ttl_seconds, max_items)
                    except sqlite3.DatabaseError:
                        LOGGER.warning("Failed to initialise HTTP cache at %s", cache_path)
        else:
            cache_ttl_days = float(self._cfg.get("cache_ttl_days", 0.0))
            cache_dir_value = self._cfg.get("cache_dir")
            if cache_dir_value and cache_ttl_days > 0:
                cache_dir = Path(cache_dir_value)
                cache_dir.mkdir(parents=True, exist_ok=True)
                ttl_seconds = int(cache_ttl_days * 86400)
                db_path = cache_dir / "http.sqlite"
                max_items_raw = self._cfg.get("cache_max_items", 0)
                try:
                    max_items = int(max_items_raw)
                except (TypeError, ValueError):
                    max_items = 0
                try:
                    self._cache = SqliteCache(str(db_path), ttl_seconds, max_items)
                except sqlite3.DatabaseError:
                    LOGGER.warning("Failed to initialise HTTP cache at %s", db_path)
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
        start = time.perf_counter()
        endpoint = self._extract_host(url) or "unknown"
        try:
            if self._cache_enabled:
                cached = self._read_cache("GET", url, b"")
                if cached:
                    request = httpx.Request("GET", url)
                    response = httpx.Response(
                        status_code=cached.status,
                        headers=cached.headers,
                        content=cached.payload,
                        request=request,
                    )
                    response.extensions["from_cache"] = True
                    duration = time.perf_counter() - start
                    LOGGER.info("GET %s status=%s duration=%.3f source=cache", url, cached.status, duration)
                    METRICS.record_cache_hit(endpoint, labels={"kind": "http"})
                    METRICS.record_http_call(
                        endpoint,
                        "GET",
                        cached.status,
                        duration,
                        labels={"kind": "http", "source": "cache"},
                    )
                    return response
                METRICS.record_cache_miss(endpoint, labels={"kind": "http"})

            if not self.allow_fetch(url, ua):
                LOGGER.warning("GET %s blocked by robots", url)
                response = self._error_response("GET", url, "blocked_by_robots")
                duration = time.perf_counter() - start
                METRICS.record_http_call(
                    endpoint,
                    "GET",
                    response.status_code,
                    duration,
                    labels={"kind": "http", "reason": "robots"},
                )
                return response

            host = self._extract_host(url)
            self._maybe_wait_sync(host)
            status_response = self._perform_sync_request("GET", url, ua)
            self._mark_sync(host)

            if status_response.status_code == httpx.codes.OK and self._cache_enabled:
                self._write_cache("GET", url, status_response, b"")
            return status_response
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("GET %s unexpected error: %s", url, exc)
            response = self._error_response("GET", url, "unexpected_error")
            duration = time.perf_counter() - start
            METRICS.record_http_call(
                endpoint,
                "GET",
                response.status_code,
                duration,
                labels={"kind": "http", "reason": "exception"},
            )
            return response

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
        start = time.perf_counter()
        endpoint = self._extract_host(url) or "unknown"
        recorded = False
        retries_used = 0
        try:
            if self._cache_enabled:
                cached = await asyncio.to_thread(self._read_cache, "GET", url, b"")
                if cached:
                    duration = time.perf_counter() - start
                    LOGGER.info("GET %s status=%s duration=%.3f source=cache", url, cached.status, duration)
                    METRICS.record_cache_hit(endpoint, labels={"kind": "http"})
                    METRICS.record_http_call(
                        endpoint,
                        "GET",
                        cached.status,
                        duration,
                        labels={"kind": "http", "source": "cache"},
                    )
                    return cached.status, cached.payload.decode("utf-8", errors="replace")
                METRICS.record_cache_miss(endpoint, labels={"kind": "http"})

            allowed = await asyncio.to_thread(self.allow_fetch, url, ua)
            if not allowed:
                LOGGER.warning("GET %s blocked by robots", url)
                duration = time.perf_counter() - start
                METRICS.record_http_call(
                    endpoint,
                    "GET",
                    0,
                    duration,
                    labels={"kind": "http", "reason": "robots"},
                )
                return 0, ""

            host = self._extract_host(url)
            semaphore = await self._per_host_semaphore(host)
            assert self._global_semaphore is not None  # sanity
            async with self._async_limit():
                async with self._global_semaphore, semaphore:
                    await self._maybe_wait_async(host)
                    attempt = 0
                    last_text = ""
                    last_status = 0
                    start_request = time.perf_counter()
                    while attempt < self._retry_attempts:
                        headers = self._build_headers(ua)
                        try:
                            response = await self._async_client.get(url, headers=headers)
                            last_status = response.status_code
                            last_text = response.text
                            duration = time.perf_counter() - start_request
                            if response.status_code in {429, 503} and attempt + 1 < self._retry_attempts:
                                LOGGER.warning(
                                    "GET %s retryable status=%s attempt=%s",
                                    url,
                                    response.status_code,
                                    attempt + 1,
                                )
                                await self._sleep_backoff_async(attempt)
                                retries_used += 1
                                attempt += 1
                                continue

                            LOGGER.info("GET %s status=%s duration=%.3f", url, response.status_code, duration)
                            METRICS.record_http_call(
                                endpoint,
                                "GET",
                                response.status_code,
                                duration,
                                retries=retries_used,
                                labels={"kind": "http"},
                            )
                            recorded = True
                            if response.status_code == httpx.codes.OK and self._cache_enabled:
                                await asyncio.to_thread(self._write_cache, "GET", url, response, b"")
                            break
                        except httpx.TimeoutException as exc:
                            duration = time.perf_counter() - start_request
                            LOGGER.warning("GET %s timeout attempt=%s duration=%.3f error=%s", url, attempt + 1, duration, exc)
                            last_status = 0
                            last_text = ""
                            if attempt + 1 >= self._retry_attempts:
                                break
                            await self._sleep_backoff_async(attempt)
                            retries_used += 1
                        except httpx.HTTPError as exc:
                            duration = time.perf_counter() - start_request
                            LOGGER.warning(
                                "GET %s http error attempt=%s duration=%.3f error=%s", url, attempt + 1, duration, exc
                            )
                            last_status = 0
                            last_text = ""
                            if attempt + 1 >= self._retry_attempts:
                                break
                            await self._sleep_backoff_async(attempt)
                            retries_used += 1
                        attempt += 1
                    await self._mark_async(host)
                    if not recorded:
                        duration = time.perf_counter() - start_request
                        METRICS.record_http_call(
                            endpoint,
                            "GET",
                            last_status,
                            duration,
                            retries=retries_used,
                            labels={"kind": "http"},
                        )
                    return last_status, last_text
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("GET %s unexpected error: %s", url, exc)
            duration = time.perf_counter() - start
            METRICS.record_http_call(
                endpoint,
                "GET",
                0,
                duration,
                labels={"kind": "http", "reason": "exception"},
            )
            return 0, ""

    def _perform_sync_request(self, method: str, url: str, ua: str) -> httpx.Response:
        headers = self._build_headers(ua)
        attempt = 0
        retries_used = 0
        last_response: Optional[httpx.Response] = None
        start = time.perf_counter()
        endpoint = self._extract_host(url) or "unknown"
        method_name = method.upper()
        while attempt < self._retry_attempts:
            try:
                with self._sync_limit():
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
                    retries_used += 1
                    attempt += 1
                    continue
                LOGGER.info("%s %s status=%s duration=%.3f", method, url, response.status_code, duration)
                METRICS.record_http_call(
                    endpoint,
                    method_name,
                    response.status_code,
                    duration,
                    retries=retries_used,
                    labels={"kind": "http"},
                )
                return response
            except httpx.TimeoutException as exc:
                duration = time.perf_counter() - start
                LOGGER.warning("%s %s timeout attempt=%s duration=%.3f error=%s", method, url, attempt + 1, duration, exc)
                if attempt + 1 >= self._retry_attempts:
                    break
                self._sleep_backoff_sync(attempt)
                retries_used += 1
            except httpx.HTTPError as exc:
                duration = time.perf_counter() - start
                LOGGER.warning("%s %s http error attempt=%s duration=%.3f error=%s", method, url, attempt + 1, duration, exc)
                if attempt + 1 >= self._retry_attempts:
                    break
                self._sleep_backoff_sync(attempt)
                retries_used += 1
            attempt += 1
        if last_response is not None:
            duration = time.perf_counter() - start
            METRICS.record_http_call(
                endpoint,
                method_name,
                last_response.status_code,
                duration,
                retries=retries_used,
                labels={"kind": "http"},
            )
            return last_response
        duration = time.perf_counter() - start
        error_response = self._error_response(method, url, "request_failed")
        METRICS.record_http_call(
            endpoint,
            method_name,
            error_response.status_code,
            duration,
            retries=retries_used,
            labels={"kind": "http", "reason": "request_failed"},
        )
        return error_response

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
            with self._sync_limit():
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

    def pick_user_agent(self) -> str:
        """Return a user agent string suitable for HTTP requests."""
        return self._choose_user_agent()

    def _choose_user_agent(self) -> str:
        return random.choice(self._user_agents)

    def _build_headers(self, ua: str) -> Dict[str, str]:
        headers = dict(self._default_headers)
        headers.setdefault("User-Agent", ua)
        headers.setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        headers.setdefault("Accept-Language", "en-US,en;q=0.5")
        return headers

    def _record_cache_lookup(self, hit: bool) -> None:
        if not self._cache_enabled:
            return
        with self._cache_stats_lock:
            self._cache_lookups += 1
            if hit:
                self._cache_hits += 1
            lookups = self._cache_lookups
            if lookups <= 10 or lookups % 100 == 0:
                ratio = self._cache_hits / lookups if lookups else 0.0
                LOGGER.info(
                    "HTTP cache hit ratio: %.2f (%d/%d)",
                    ratio,
                    self._cache_hits,
                    lookups,
                )

    def _read_cache(self, method: str, url: str, body: bytes) -> Optional[CachedResponse]:
        if not self._cache_enabled or self._cache is None:
            return None
        params_hash = SqliteCache.hash_payload(body or b"")
        cached = self._cache.get(url, method, params_hash)
        self._record_cache_lookup(cached is not None)
        return cached

    def _write_cache(self, method: str, url: str, response: httpx.Response, body: bytes) -> None:
        if not self._cache_enabled or self._cache is None:
            return
        params_hash = SqliteCache.hash_payload(body or b"")
        headers = dict(response.headers.items())
        self._cache.set(
            url=url,
            method=method,
            params_hash=params_hash,
            payload=response.content,
            headers=headers,
            status=response.status_code,
        )

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

    @contextmanager
    def _sync_limit(self) -> Iterable[None]:
        if not self._shared_limiter:
            yield
        else:
            with self._shared_limiter.sync():
                yield

    @asynccontextmanager
    async def _async_limit(self) -> Iterable[None]:
        if not self._shared_limiter:
            yield
        else:
            async with self._shared_limiter.async_acquire():
                yield

    def _extract_host(self, url: str) -> str:
        parsed = urlparse(url)
        return (parsed.hostname or "").lower()

    def _error_response(self, method: str, url: str, reason: str) -> httpx.Response:
        request = httpx.Request(method, url)
        content = reason.encode("utf-8")
        return httpx.Response(status_code=0, request=request, content=content, reason=reason)
