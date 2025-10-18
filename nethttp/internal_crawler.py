"""Utility to crawl internal pages of a website starting from a homepage."""
from __future__ import annotations

import heapq
import itertools
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

__all__ = ["crawl_internal_pages", "CrawlResult", "PageResult"]

_DEFAULT_KEYWORDS = (
    "contact",
    "contacts",
    "equipe",
    "équipe",
    "a-propos",
    "apropos",
    "a_propos",
    "about",
    "mentions",
    "mentions-legales",
    "mentions_légales",
    "mentions légales",
    "legal",
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@dataclass(frozen=True)
class PageResult:
    """Result of a single crawled page."""

    url: str
    text: str
    emails: Tuple[str, ...]


@dataclass(frozen=True)
class CrawlResult:
    """Result of a crawl, including all visited pages and aggregated emails."""

    pages: Tuple[PageResult, ...]
    emails: Tuple[str, ...]


def crawl_internal_pages(
    homepage: str,
    max_pages: int,
    *,
    keywords: Optional[Sequence[str]] = None,
    client: Optional[httpx.Client] = None,
    timeout: Optional[float] = 10.0,
) -> CrawlResult:
    """Crawl internal pages starting from ``homepage``.

    Parameters
    ----------
    homepage:
        URL of the homepage to start crawling from.
    max_pages:
        Maximum number of pages to crawl. Values less than or equal to zero
        will result in an empty crawl.
    keywords:
        Optional sequence of keywords to prioritise in URLs. When omitted, a
        default list targeting typical contact/about pages is used.
    client:
        Optional :class:`httpx.Client` instance. If provided it is not closed by
        the crawler. When omitted a short-lived client is created.
    timeout:
        Optional timeout in seconds to use for HTTP requests.

    Returns
    -------
    CrawlResult
        A dataclass containing the crawled pages along with all emails
        extracted across the crawl.
    """

    if max_pages <= 0:
        return CrawlResult(pages=tuple(), emails=tuple())

    parsed_home = urlparse(homepage)
    if not parsed_home.scheme or not parsed_home.netloc:
        raise ValueError("homepage must be a valid absolute URL")

    homepage = _normalise_url(homepage)
    keywords = tuple(kw.lower() for kw in (keywords or _DEFAULT_KEYWORDS))
    allowed_hosts = _allowed_hosts(parsed_home.hostname or "")

    enqueued: Set[str] = set()
    visited: Set[str] = set()
    aggregated_emails: Set[str] = set()
    pages: List[PageResult] = []

    counter = itertools.count()
    queue: List[Tuple[Tuple[int, int], int, str]] = []

    def enqueue(url: str) -> None:
        normalised = _normalise_url(url)
        if not normalised:
            return
        parsed = urlparse(normalised)
        if parsed.hostname and parsed.hostname.lower() not in allowed_hosts:
            return
        if normalised in enqueued:
            return
        enqueued.add(normalised)
        priority = _priority(normalised, keywords, homepage)
        heapq.heappush(queue, (priority, next(counter), normalised))

    enqueue(homepage)

    owns_client = client is None
    if owns_client:
        client = httpx.Client(follow_redirects=True)
    assert client is not None  # for mypy

    try:
        while queue and len(pages) < max_pages:
            _, _, current_url = heapq.heappop(queue)
            if current_url in visited:
                continue
            visited.add(current_url)

            try:
                response = client.get(
                    current_url,
                    timeout=timeout,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; InternalCrawler/1.0)"},
                )
            except httpx.HTTPError:
                continue

            if response.status_code != 200:
                continue

            content_type = response.headers.get("content-type", "").lower()
            if "html" not in content_type:
                continue

            html = response.text
            text = _extract_text(html)
            emails = tuple(sorted(set(_extract_emails(text))))
            aggregated_emails.update(emails)
            pages.append(PageResult(url=current_url, text=text, emails=emails))

            for link in _extract_links(current_url, html):
                enqueue(link)
    finally:
        if owns_client:
            client.close()

    return CrawlResult(pages=tuple(pages), emails=tuple(sorted(aggregated_emails)))


def _priority(url: str, keywords: Sequence[str], homepage: str) -> Tuple[int, int]:
    if url == homepage:
        return (0, 0)
    path = urlparse(url).path.lower()
    for idx, keyword in enumerate(keywords):
        if keyword and keyword in path:
            return (1, idx)
    return (2, 0)


def _extract_links(base_url: str, html: str) -> Iterable[str]:
    soup = BeautifulSoup(html, "lxml")
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        if not href:
            continue
        if href.startswith(("javascript:", "mailto:")):
            continue
        absolute = urljoin(base_url, href)
        normalised = _normalise_url(absolute)
        if normalised:
            yield normalised


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    return text


def _extract_emails(text: str) -> Set[str]:
    return {match.group(0) for match in _EMAIL_RE.finditer(text)}


def _allowed_hosts(hostname: str) -> Set[str]:
    hostname = hostname.lower()
    if not hostname:
        return set()
    allowed = {hostname}
    if hostname.startswith("www."):
        allowed.add(hostname[4:])
    else:
        allowed.add(f"www.{hostname}")
    return allowed


def _normalise_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    # Remove fragments and normalise scheme/host case.
    fragmentless = parsed._replace(fragment="")
    # Normalise default ports.
    netloc = fragmentless.netloc
    if fragmentless.scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif fragmentless.scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]
    fragmentless = fragmentless._replace(netloc=netloc)
    return urlunparse(
        (
            fragmentless.scheme.lower(),
            fragmentless.netloc.lower(),
            fragmentless.path,
            fragmentless.params,
            fragmentless.query,
            "",
        )
    )
