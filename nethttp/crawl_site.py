"""Targeted crawl of official websites discovered through SERP."""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from readability import Document

from utils import io
from utils.rate import PerHostRateLimiter, TimeBudget
from utils.robots import RobotsCache
from utils.ua import UserAgentPool, load_user_agent_pool
from utils.url import canonicalize, looks_like_home, registered_domain, resolve, strip_fragment

MAX_PAGE_BYTES = 1_000_000
HTML_TRUNCATE_CHARS = 100_000
TARGET_PATHS = [
    "/mentions-legales",
    "/mentions-légales",
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


@dataclass
class CrawlTarget:
    domain: str
    start_urls: List[str]
    sirens: Set[str]
    denominations: Set[str]


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


def run(cfg: dict, ctx: dict) -> dict:
    logger = ctx.get("logger")
    outdir = Path(ctx["outdir"])
    serp_path = outdir / "serp" / "serp_results.parquet"
    if not serp_path.exists():
        if logger:
            logger.warning("crawl_site: serp results not found at %s", serp_path)
        return {"status": "SKIPPED", "reason": "NO_SERP"}

    crawl_cfg = (cfg.get("crawl") or {})
    max_pages = int(ctx.get("max_pages_per_domain") or crawl_cfg.get("max_pages_per_domain") or 12)
    per_host_rps = float(crawl_cfg.get("per_host_rps") or 1.0)
    respect_robots = bool(ctx.get("respect_robots", True)) if ctx.get("respect_robots") is not None else bool(crawl_cfg.get("respect_robots", True))
    time_budget = TimeBudget(ctx.get("crawl_time_budget_min") or crawl_cfg.get("time_budget_min"))
    timeout = float(crawl_cfg.get("timeout_sec") or DEFAULT_TIMEOUT)

    targets = _load_targets(serp_path)
    sample = int(ctx.get("sample") or 0)
    if sample > 0:
        targets = targets[:sample]
    elif ctx.get("dry_run"):
        targets = targets[: min(5, len(targets))]

    if not targets:
        return {"status": "SKIPPED", "reason": "NO_TARGETS"}

    ua_pool: UserAgentPool = ctx.get("user_agent_pool") or load_user_agent_pool(None)
    crawler_ua = ctx.get("crawler_user_agent") or DEFAULT_CRAWLER_UA
    robots_cache = RobotsCache(user_agent=crawler_ua)
    limiter = PerHostRateLimiter(per_host_rps=per_host_rps, jitter_range=(0.2, 0.8))

    pages: List[Dict[str, object]] = []

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for target in targets:
            if time_budget.exhausted:
                if logger:
                    logger.info("crawl_site: time budget exhausted")
                break
            base_domain = target.domain
            queue: List[tuple[int, int, str]] = []
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

            while queue and pages_crawled < max_pages and not time_budget.exhausted:
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
                if respect_robots and not robots_cache.allowed(current_url, respect_robots=True):
                    if logger:
                        logger.debug("crawl_site: blocked by robots %s", current_url)
                    continue
                crawl_delay = robots_cache.crawl_delay(current_url) if respect_robots else None
                if crawl_delay and crawl_delay > 0:
                    time.sleep(crawl_delay)
                limiter.wait(host)
                headers = {
                    "User-Agent": ua_pool.get(),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.6",
                }
                try:
                    response = client.get(current_url, headers=headers)
                except httpx.HTTPError as exc:
                    if logger:
                        logger.debug("crawl_site: request failed %s: %s", current_url, exc)
                    continue

                final_url = strip_fragment(str(response.url))
                status = response.status_code
                content_type = (response.headers.get("content-type") or "").split(";")[0].lower()
                discovered_at = io.now_iso()
                content_bytes = response.content or b""
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

    if not pages:
        return {"status": "SKIPPED", "reason": "NO_PAGES"}

    out_dir = io.ensure_dir(outdir / "crawl")
    output_path = out_dir / "pages.parquet"
    pd.DataFrame(pages).to_parquet(output_path, index=False)

    return {
        "status": "OK",
        "output": str(output_path),
        "pages": len(pages),
    }
