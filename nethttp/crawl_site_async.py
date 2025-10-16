"""Async targeted crawl of official websites discovered through SERP."""
from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from readability import Document

from metrics.collector import get_metrics
from utils import budget_middleware, io
from utils.rate import PerHostRateLimiter, TimeBudget
from utils.robots import RobotsCache
from utils.ua import UserAgentPool, load_user_agent_pool
from utils.url import canonicalize, looks_like_home, registered_domain, resolve, strip_fragment
from utils.state import SequentialRunState

METRICS = get_metrics()

MAX_PAGE_BYTES = 1_000_000
HTML_TRUNCATE_CHARS = 100_000
TARGET_PATHS = [
    "/mentions-legales",
    "/mentions-l\u00e9gales",
    "/mentions\xa0legales",
    "/legal",
    "/contact",
    "/a-propos",
    "/a_propos",
    "/about",
    "/.well-known/security.txt",
]
SKIP_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".mp4",
    ".mp3",
    ".avi",
    ".mov",
    ".mkv",
    ".zip",
    ".rar",
    ".7z",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
}
DEFAULT_CRAWLER_UA = "Mozilla/5.0 (compatible; IM-DataEnricher/0.1; +https://projects.infinitemoula.fr/robots)"
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_DOMAINS_PARALLEL = 10


@dataclass
class CrawlTarget:
    domain: str
    start_urls: List[str]
    sirens: Set[str]
    denominations: Set[str]


class AsyncPerHostRateLimiter:
    """Async wrapper around the synchronous `PerHostRateLimiter`."""

    def __init__(self, per_host_rps: float = 1.0, jitter_range: Tuple[float, float] = (0.2, 0.8)) -> None:
        self._limiter = PerHostRateLimiter(per_host_rps=per_host_rps, jitter_range=jitter_range)

    async def wait(self, host: str) -> float:
        if not host:
            return 0.0
        return await asyncio.to_thread(self._limiter.wait, host)


def _load_targets(serp_path: Path) -> List[CrawlTarget]:
    df = pd.read_parquet(serp_path)
    grouped: Dict[str, CrawlTarget] = {}
    for _, row in df.iterrows():
        top_url = str(row.get("top_url") or "").strip()
        top_domain = str(row.get("top_domain") or "").strip()
        if not top_domain and top_url:
            top_domain = registered_domain(top_url)
        if not top_domain:
            continue
        entry = grouped.get(top_domain)
        if not entry:
            entry = CrawlTarget(domain=top_domain, start_urls=[], sirens=set(), denominations=set())
            grouped[top_domain] = entry
        if top_url:
            entry.start_urls.append(top_url)
        else:
            entry.start_urls.append(f"https://{top_domain}")
        siren = str(row.get("siren") or "").strip()
        if siren:
            entry.sirens.add(siren)
        denomination = str(row.get("denomination") or "").strip()
        if denomination:
            entry.denominations.add(denomination)
    return list(grouped.values())


def _path_priority(url: str) -> int:
    parsed = urlparse(url)
    path = parsed.path.lower()
    if looks_like_home(url):
        return 1
    for idx, target in enumerate(TARGET_PATHS):
        if path.startswith(target):
            return idx
    if path.endswith("/"):
        return len(TARGET_PATHS) + 1
    return len(TARGET_PATHS) + 5


def _should_skip(url: str, base_domain: str, origin_host: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return True
    host = (parsed.hostname or "").lower()
    allowed_hosts = {
        origin_host.lower(),
        base_domain.lower(),
        f"www.{base_domain.lower()}",
    }
    if origin_host.lower().startswith("www."):
        allowed_hosts.add(origin_host.lower()[4:])
    if host and host not in allowed_hosts:
        return True
    path = parsed.path.lower()
    for ext in SKIP_EXTENSIONS:
        if path.endswith(ext):
            return True
    return False


def _extract_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links: List[str] = []
    for link in soup.find_all("a", href=True):
        href = link.get("href")
        if href and not href.startswith(("javascript:", "mailto:")):
            resolved = resolve(base_url, href)
            if resolved:
                links.append(strip_fragment(resolved))
    return links


def _extract_text(html: str) -> str:
    try:
        document = Document(html)
        summary_html = document.summary()
        soup = BeautifulSoup(summary_html, "lxml")
        text = soup.get_text(" ", strip=True)
    except Exception:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(" ", strip=True)
    return text[:200000]


async def _robots_allowed(robots_cache: RobotsCache, url: str, respect_robots: bool) -> bool:
    if not respect_robots:
        return True
    return await asyncio.to_thread(robots_cache.allowed, url, True)


async def _crawl_delay(robots_cache: RobotsCache, url: str, respect_robots: bool) -> Optional[float]:
    if not respect_robots:
        return None
    return await asyncio.to_thread(robots_cache.crawl_delay, url)


async def _crawl_target(
    target: CrawlTarget,
    *,
    client: httpx.AsyncClient,
    ua_pool: UserAgentPool,
    robots_cache: RobotsCache,
    limiter: AsyncPerHostRateLimiter,
    logger,
    max_pages: int,
    respect_robots: bool,
    time_budget: TimeBudget,
    request_tracker: Optional[Callable[[int], None]] = None,
) -> Tuple[str, List[Dict[str, object]], Dict[str, int], Dict[str, object]]:
    pages: List[Dict[str, object]] = []
    stats = {"errors": 0, "status_4xx": 0, "status_5xx": 0}

    base_domain = target.domain
    queue: List[Tuple[int, int, str]] = []
    visited: Set[str] = set()
    enqueued: Set[str] = set()
    counter = itertools.count()

    start_urls = [canonicalize(url) for url in dict.fromkeys(target.start_urls)]
    if not start_urls:
        start_urls = [f"https://{base_domain}"]
    origin_host = urlparse(start_urls[0]).hostname or base_domain
    for url in start_urls:
        if url in enqueued:
            continue
        heapq.heappush(queue, (_path_priority(url), next(counter), url))
        enqueued.add(url)

    pages_crawled = 0

    while queue and pages_crawled < max_pages:
        if time_budget.exhausted:
            if logger:
                logger.debug("crawl_site_async: time budget exhausted during crawl of %s", base_domain)
            break

        _, _, current_url = heapq.heappop(queue)
        if current_url in visited:
            continue
        visited.add(current_url)

        parsed = urlparse(current_url)
        host = (parsed.hostname or "").lower()
        if not host:
            continue
        if _should_skip(current_url, base_domain, origin_host):
            continue
        allowed = await _robots_allowed(robots_cache, current_url, respect_robots)
        if not allowed:
            if logger:
                logger.debug("crawl_site_async: blocked by robots %s", current_url)
            continue

        crawl_delay = await _crawl_delay(robots_cache, current_url, respect_robots)
        if crawl_delay and crawl_delay > 0:
            await asyncio.sleep(crawl_delay)

        await limiter.wait(host)

        headers = {
            "User-Agent": ua_pool.get(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.6",
        }

        request_start = time.perf_counter()
        try:
            response = await client.get(current_url, headers=headers)
            duration = time.perf_counter() - request_start
            METRICS.record_http_call(
                host,
                "GET",
                response.status_code,
                duration,
                labels={"kind": "crawler", "group": f"crawler:{host}"},
            )
        except httpx.HTTPError as exc:
            duration = time.perf_counter() - request_start
            METRICS.record_http_call(
                host,
                "GET",
                0,
                duration,
                labels={
                    "kind": "crawler",
                    "reason": "http_error",
                    "group": f"crawler:{host}",
                },
            )
            stats["errors"] += 1
            if request_tracker:
                try:
                    request_tracker(0)
                except budget_middleware.BudgetExceededError:
                    raise
            if logger:
                logger.debug("crawl_site_async: request failed %s: %s", current_url, exc)
            continue

        final_url = strip_fragment(str(response.url))
        status = response.status_code
        content_type = (response.headers.get("content-type") or "").split(";")[0].lower()
        discovered_at = io.now_iso()
        content_bytes = response.content or b""
        try:
            if request_tracker:
                request_tracker(len(content_bytes))
        except budget_middleware.BudgetExceededError:
            raise

        page_entry = {
            "domain": base_domain,
            "requested_url": current_url,
            "url": final_url,
            "status": status,
            "content_type": content_type,
            "bytes": len(content_bytes),
            "discovered_at": discovered_at,
            "siren_list": sorted(target.sirens),
            "denominations": sorted(target.denominations),
            "https": final_url.startswith("https"),
        }

        if 400 <= status < 500:
            stats["status_4xx"] += 1
        elif status >= 500:
            stats["status_5xx"] += 1

        if status >= 400:
            pages.append(page_entry)
            continue
        if "text" not in content_type and "json" not in content_type:
            pages.append(page_entry)
            continue
        if len(content_bytes) > MAX_PAGE_BYTES:
            content_bytes = content_bytes[:MAX_PAGE_BYTES]
        try:
            html = content_bytes.decode(response.encoding or "utf-8", errors="replace")
        except LookupError:
            html = content_bytes.decode("utf-8", errors="replace")
        html_trunc = html[:HTML_TRUNCATE_CHARS]
        text_content = _extract_text(html)
        page_entry.update(
            {
                "content_text": text_content,
                "content_html_trunc": html_trunc,
            }
        )
        pages.append(page_entry)
        pages_crawled += 1

        if "html" in content_type and pages_crawled < max_pages:
            for link in _extract_links(final_url, html):
                if link in enqueued:
                    continue
                if _should_skip(link, base_domain, origin_host):
                    continue
                heapq.heappush(queue, (_path_priority(link), next(counter), link))
                enqueued.add(link)

    metadata = {
        "pages_crawled": pages_crawled,
        "visited_urls": len(visited),
        "start_urls": start_urls,
        "incomplete": bool(queue),
    }
    return base_domain, pages, stats, metadata


async def _crawl_targets_async(
    targets: List[CrawlTarget],
    *,
    client: httpx.AsyncClient,
    ua_pool: UserAgentPool,
    robots_cache: RobotsCache,
    limiter: AsyncPerHostRateLimiter,
    logger,
    max_pages: int,
    respect_robots: bool,
    time_budget: TimeBudget,
    max_domains_parallel: int,
    request_tracker: Optional[Callable[[int], None]] = None,
    state: Optional[SequentialRunState] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, int], float]:
    pages: List[Dict[str, object]] = []
    stats = {"errors": 0, "status_4xx": 0, "status_5xx": 0}
    semaphore = asyncio.Semaphore(max(1, max_domains_parallel))
    start = time.perf_counter()

    if state:
        for target in targets:
            state.mark_started(target.domain)

    async def _bounded_crawl(
        target: CrawlTarget,
    ) -> Tuple[str, List[Dict[str, object]], Dict[str, int], Dict[str, object], Optional[str]]:
        async with semaphore:
            if time_budget.exhausted:
                return (
                    target.domain,
                    [],
                    {"errors": 0, "status_4xx": 0, "status_5xx": 0},
                    {
                        "pages_crawled": 0,
                        "visited_urls": 0,
                        "start_urls": list(target.start_urls),
                        "incomplete": True,
                    },
                    "time_budget_exhausted",
                )
            try:
                domain, domain_pages, domain_stats, domain_meta = await _crawl_target(
                    target,
                    client=client,
                    ua_pool=ua_pool,
                    robots_cache=robots_cache,
                    limiter=limiter,
                    logger=logger,
                    max_pages=max_pages,
                    respect_robots=respect_robots,
                    time_budget=time_budget,
                    request_tracker=request_tracker,
                )
            except budget_middleware.BudgetExceededError:
                raise
            except Exception as exc:
                if logger:
                    logger.warning("crawl_site_async: domain %s failed: %s", target.domain, exc)
                return (
                    target.domain,
                    [],
                    {"errors": 1, "status_4xx": 0, "status_5xx": 0},
                    {
                        "pages_crawled": 0,
                        "visited_urls": 0,
                        "start_urls": list(target.start_urls),
                        "incomplete": True,
                    },
                    str(exc),
                )
            return domain, domain_pages, domain_stats, domain_meta, None

    tasks = [asyncio.create_task(_bounded_crawl(target)) for target in targets]
    pending = set(tasks)
    try:
        for task in asyncio.as_completed(tasks):
            domain, domain_pages, domain_stats, domain_meta, domain_error = await task
            pending.discard(task)
            pages.extend(domain_pages)
            stats["errors"] += domain_stats.get("errors", 0)
            stats["status_4xx"] += domain_stats.get("status_4xx", 0)
            stats["status_5xx"] += domain_stats.get("status_5xx", 0)
            if state:
                if domain_error:
                    state.mark_failed(domain, domain_error)
                elif domain_meta.get("incomplete") and time_budget.exhausted:
                    state.mark_failed(domain, "time_budget_exhausted")
                else:
                    state.mark_completed(domain, extra=domain_meta)
            if time_budget.exhausted and domain_meta.get("incomplete"):
                break
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    elapsed = time.perf_counter() - start
    return pages, stats, elapsed


def run(cfg: dict, ctx: dict) -> dict:
    logger = ctx.get("logger")
    outdir = Path(ctx["outdir"])
    serp_path = outdir / "serp" / "serp_results.parquet"
    if not serp_path.exists():
        if logger:
            logger.warning("crawl_site_async: serp results not found at %s", serp_path)
        return {"status": "SKIPPED", "reason": "NO_SERP"}

    crawl_cfg = cfg.get("crawl", {})
    max_pages = int(ctx.get("max_pages_per_domain") or crawl_cfg.get("max_pages_per_domain") or 12)
    per_host_rps = float(ctx.get("per_host_rps") or crawl_cfg.get("per_host_rps") or 1.0)
    respect_robots = bool(ctx.get("respect_robots", True)) if ctx.get("respect_robots") is not None else bool(
        crawl_cfg.get("respect_robots", True)
    )
    time_budget = TimeBudget(ctx.get("crawl_time_budget_min") or crawl_cfg.get("time_budget_min"))
    timeout = float(
        ctx.get("crawl_timeout_s")
        or crawl_cfg.get("timeout_s")
        or crawl_cfg.get("timeout_sec")
        or DEFAULT_TIMEOUT
    )
    max_domains_parallel = int(
        ctx.get("max_domains_parallel")
        or crawl_cfg.get("max_domains_parallel")
        or DEFAULT_MAX_DOMAINS_PARALLEL
    )

    targets = _load_targets(serp_path)
    sample = int(ctx.get("sample") or 0)
    if sample > 0:
        targets = targets[:sample]
    elif ctx.get("dry_run"):
        targets = targets[: min(5, len(targets))]

    if not targets:
        return {"status": "SKIPPED", "reason": "NO_TARGETS"}

    crawl_dir = io.ensure_dir(outdir / "crawl")
    state = SequentialRunState(crawl_dir / "crawl_async_state.json")
    target_map = {target.domain: target for target in targets}
    state.set_metadata(total=len(target_map))
    pending_domains = set(state.pending(target_map.keys()))
    selected_targets = [target for target in targets if target.domain in pending_domains]

    existing_output = crawl_dir / "pages.parquet"
    existing_pages: List[Dict[str, object]] = []
    if existing_output.exists():
        try:
            existing_df = pd.read_parquet(existing_output)
            existing_pages = existing_df.to_dict(orient="records")
        except Exception:
            if logger:
                logger.debug("crawl_site_async: unable to preload existing pages from %s", existing_output)

    if not selected_targets:
        if existing_pages:
            state.set_metadata(last_output=str(existing_output), last_pages=len(existing_pages))
            return {
                "status": "SKIPPED",
                "reason": "NO_PENDING_TARGETS",
                "output": str(existing_output),
                "pages": len(existing_pages),
            }
        return {"status": "SKIPPED", "reason": "NO_PENDING_TARGETS"}

    ua_pool: UserAgentPool = ctx.get("user_agent_pool") or load_user_agent_pool(None)
    crawler_ua = ctx.get("crawler_user_agent") or DEFAULT_CRAWLER_UA
    robots_cache = RobotsCache(user_agent=crawler_ua)
    limiter = AsyncPerHostRateLimiter(per_host_rps=per_host_rps, jitter_range=(0.2, 0.8))
    request_tracker = ctx.get("request_tracker")

    async def _execute() -> Tuple[List[Dict[str, object]], Dict[str, int], float]:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            return await _crawl_targets_async(
                selected_targets,
                client=client,
                ua_pool=ua_pool,
                robots_cache=robots_cache,
                limiter=limiter,
                logger=logger,
                max_pages=max_pages,
                respect_robots=respect_robots,
                time_budget=time_budget,
                max_domains_parallel=max_domains_parallel,
                request_tracker=request_tracker,
                state=state,
            )

    new_pages, stats, elapsed = asyncio.run(_execute())

    combined_pages = existing_pages + new_pages

    if not combined_pages:
        return {"status": "SKIPPED", "reason": "NO_PAGES"}

    output_path = crawl_dir / "pages.parquet"
    pd.DataFrame(combined_pages).to_parquet(output_path, index=False)
    state.set_metadata(last_output=str(output_path), last_pages=len(combined_pages))

    pages_per_sec = (len(new_pages) / elapsed) if elapsed > 0 else 0.0
    if logger:
        logger.info(
            "crawl_site_async: %d pages in %.2fs (%.2f pages/s) errors=%d 4xx=%d 5xx=%d",
            len(new_pages),
            elapsed,
            pages_per_sec,
            stats.get("errors", 0),
            stats.get("status_4xx", 0),
            stats.get("status_5xx", 0),
        )

    return {
        "status": "OK",
        "output": str(output_path),
        "pages": len(combined_pages),
    }
