"""Scrape email addresses from a web page, including dynamic content."""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Coroutine, Optional, Set

import httpx

try:  # pragma: no cover - optional dependency
    from playwright.async_api import (  # type: ignore
        Browser,
        Error as PlaywrightError,
        TimeoutError as PlaywrightTimeoutError,
        async_playwright,
    )
except Exception:  # pragma: no cover - fallback when Playwright is unavailable
    Browser = object  # type: ignore[misc,assignment]
    PlaywrightError = RuntimeError  # type: ignore[misc,assignment]
    PlaywrightTimeoutError = RuntimeError  # type: ignore[misc,assignment]
    async_playwright = None  # type: ignore[misc,assignment]

LOGGER = logging.getLogger("scraper.email_scraper")

_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Z0-9-]+\.)+[A-Z]{2,63}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EmailScrapeResult:
    """Result of an email scraping attempt."""

    emails: Set[str]
    source: str


def _extract_emails(text: str) -> Set[str]:
    """Return a set of normalized email addresses found in *text*."""

    if not text:
        return set()
    return {match.group(0).lower() for match in _EMAIL_PATTERN.finditer(text)}


def _fetch_static_html(
    url: str,
    *,
    timeout: float,
    headers: Optional[dict[str, str]] = None,
) -> str:
    """Fetch *url* using httpx and return the response text."""

    request_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0 Safari/537.36"
        )
    }
    if headers:
        request_headers.update(headers)

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url, headers=request_headers)
        response.raise_for_status()
        return response.text


async def _render_with_playwright(
    url: str,
    *,
    timeout: float,
    headers: Optional[dict[str, str]] = None,
    wait_until: str = "networkidle",
    extra_wait: float = 0.0,
) -> str:
    """Render *url* with Playwright and return the combined HTML and text."""

    if async_playwright is None:  # pragma: no cover - runtime safeguard
        raise RuntimeError("Playwright is not available in this environment")

    playwright = await async_playwright().start()
    browser: Browser | None = None
    context = None
    page = None
    try:
        launch_headers = dict(headers or {})
        user_agent = launch_headers.pop("User-Agent", None)
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=user_agent,
            extra_http_headers=launch_headers or None,
        )
        page = await context.new_page()
        await page.goto(url, wait_until=wait_until, timeout=int(timeout * 1000))
        if extra_wait:
            await page.wait_for_timeout(int(extra_wait * 1000))
        html = await page.content()
        text = await page.evaluate("() => document.body ? document.body.innerText : ''")
        return f"{html}\n{text}"
    finally:
        if page is not None:
            await page.close()
        if context is not None:
            await context.close()
        if browser is not None:
            await browser.close()
        await playwright.stop()


def _run_async(coro: Coroutine[Any, Any, str]) -> str:
    """Run *coro* in a new event loop if necessary."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError(
            "Cannot run Playwright rendering inside a running event loop; "
            "use the asynchronous scraping helpers instead."
        )

    return asyncio.run(coro)


def scrape_emails_from_url(
    url: str,
    *,
    request_timeout: float = 10.0,
    playwright_timeout: float = 15.0,
    headers: Optional[dict[str, str]] = None,
    wait_until: str = "networkidle",
    extra_wait: float = 0.0,
) -> EmailScrapeResult:
    """Scrape *url* and return extracted email addresses.

    The function first performs a classic HTTP GET request and extracts emails
    from the static HTML payload. If no address is found, it retries the same
    URL with Playwright to render JavaScript-driven content.
    """

    try:
        html = _fetch_static_html(url, timeout=request_timeout, headers=headers)
    except httpx.HTTPError as exc:
        LOGGER.warning("Static fetch failed for %s: %s", url, exc)
        html = ""

    emails = _extract_emails(html)
    if emails:
        return EmailScrapeResult(emails=emails, source="static")

    LOGGER.info("No email found in static content for %s; attempting Playwright", url)

    try:
        rendered = _run_async(
            _render_with_playwright(
                url,
                timeout=playwright_timeout,
                headers=headers,
                wait_until=wait_until,
                extra_wait=extra_wait,
            )
        )
    except RuntimeError as exc:
        # Either Playwright is missing or we cannot create a new event loop
        LOGGER.warning("Playwright execution skipped for %s: %s", url, exc)
        return EmailScrapeResult(emails=set(), source="static")
    except PlaywrightTimeoutError as exc:
        LOGGER.warning("Playwright timed out for %s: %s", url, exc)
        return EmailScrapeResult(emails=set(), source="playwright")
    except PlaywrightError as exc:
        LOGGER.warning("Playwright failed for %s: %s", url, exc)
        return EmailScrapeResult(emails=set(), source="playwright")

    emails = _extract_emails(rendered)
    return EmailScrapeResult(emails=emails, source="playwright")


__all__ = ["EmailScrapeResult", "scrape_emails_from_url"]
