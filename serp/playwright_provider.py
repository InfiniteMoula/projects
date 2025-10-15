from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Coroutine, Iterable, List, Mapping, Sequence, Tuple

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import Browser, Page, Playwright, TimeoutError as PlaywrightTimeoutError, async_playwright

from serp.providers import Result, SerpProvider

LOGGER = logging.getLogger("serp.playwright_provider")


_DEFAULT_USER_AGENTS: Tuple[str, ...] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36",
)


@dataclass(slots=True)
class _RawCard:
    url: str
    title: str
    snippet: str


class PlaywrightSerpProvider(SerpProvider):
    """
    Base class wrapping Playwright to render SERP pages when static HTML endpoints stop working.
    """

    ENGINE: str = ""
    WAIT_FOR_SELECTOR: str = ""

    def __init__(self, cfg: Mapping[str, Any], http_client) -> None:
        super().__init__(cfg, http_client)
        self._headless = bool(self._cfg.get("headless", True))
        self._locale = str(self._cfg.get("locale", "fr-FR"))
        self._viewport = self._sanitize_viewport(self._cfg.get("viewport"))
        self._user_agents = self._normalize_user_agents(self._cfg.get("user_agents"))
        self._navigation_timeout_ms = max(1_000, int(self._cfg.get("navigation_timeout_ms", 15_000)))
        self._wait_after_navigation_ms = max(0, int(self._cfg.get("wait_after_navigation_ms", 1_000)))
        self._wait_until = str(self._cfg.get("wait_until", "domcontentloaded"))
        self._launch_args = tuple(str(arg) for arg in self._cfg.get("launch_args", ()))
        self._extra_headers = {
            str(key): str(value)
            for key, value in dict(self._cfg.get("extra_http_headers", {})).items()
            if key and value is not None
        }

    @staticmethod
    def _sanitize_viewport(value: Any) -> Mapping[str, int]:
        if not isinstance(value, Mapping):
            return {"width": 1280, "height": 720}
        try:
            width = int(value.get("width", 1280))
            height = int(value.get("height", 720))
            return {"width": width, "height": height}
        except (TypeError, ValueError):
            return {"width": 1280, "height": 720}

    @staticmethod
    def _normalize_user_agents(value: Any) -> Tuple[str, ...]:
        cleaned: List[str] = []
        if isinstance(value, (list, tuple, set)):
            for ua in value:
                text = str(ua).strip()
                if text:
                    cleaned.append(text)
        if not cleaned:
            cleaned = list(_DEFAULT_USER_AGENTS)
        return tuple(cleaned)

    def search(self, query: str) -> List[Result]:
        try:
            return self._run_sync(self._search_async(query))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Playwright %s provider crashed for %r: %s", self.ENGINE, query, exc)
            return []

    def _run_sync(self, coro: Coroutine[Any, Any, List[Result]]) -> List[Result]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    async def _search_async(self, query: str) -> List[Result]:
        user_agent = random.choice(self._user_agents)
        search_url = self._build_url(query)
        html = ""
        base_url = search_url
        playwright: Playwright | None = None
        browser: Browser | None = None
        context = None
        page: Page | None = None

        try:
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=self._headless, args=self._launch_args)
            context = await browser.new_context(
                user_agent=user_agent,
                locale=self._locale,
                viewport=self._viewport,
                ignore_https_errors=bool(self._cfg.get("ignore_https_errors", False)),
            )
            if self._extra_headers:
                await context.set_extra_http_headers(self._extra_headers)

            page = await context.new_page()
            await page.goto(search_url, wait_until=self._wait_until, timeout=self._navigation_timeout_ms)
            await self._after_navigation(page)
            if self._wait_after_navigation_ms:
                await page.wait_for_timeout(self._wait_after_navigation_ms)
            if self.WAIT_FOR_SELECTOR:
                await page.wait_for_selector(self.WAIT_FOR_SELECTOR, timeout=self._navigation_timeout_ms)
            html = await page.content()
            base_url = page.url
        except PlaywrightTimeoutError:
            LOGGER.warning("Playwright %s timed out for query %r", self.ENGINE, query)
            return []
        except Exception as exc:
            LOGGER.warning("Playwright %s failed for %r: %s", self.ENGINE, query, exc)
            return []
        finally:
            if page is not None:
                await page.close()
            if context is not None:
                await context.close()
            if browser is not None:
                await browser.close()
            if playwright is not None:
                await playwright.stop()

        return self._parse_results(html, base_url)

    def _parse_results(self, html: str, base_url: str) -> List[Result]:
        soup = BeautifulSoup(html, "lxml")
        raw_cards = self._extract_cards(soup, base_url)
        results: List[Result] = []
        for raw in raw_cards:
            cleaned_url = self._clean_url(raw.url, base=base_url)
            if not cleaned_url:
                continue
            domain = self._extract_domain(cleaned_url)
            if not domain or self._is_generic_domain(domain):
                continue
            results.append(
                Result(
                    url=cleaned_url,
                    domain=domain,
                    title=raw.title,
                    snippet=raw.snippet,
                    rank=len(results) + 1,
                )
            )
            if len(results) >= self.max_results:
                break
        return results

    async def _after_navigation(self, page: Page) -> None:
        return

    def _build_url(self, query: str) -> str:  # pragma: no cover - abstract behaviour
        raise NotImplementedError

    def _extract_cards(self, soup: BeautifulSoup, base_url: str) -> Sequence[_RawCard]:  # pragma: no cover - abstract
        raise NotImplementedError


class PlaywrightBingProvider(PlaywrightSerpProvider):
    ENGINE = "bing"
    WAIT_FOR_SELECTOR = "li.b_algo h2 a"

    def _build_url(self, query: str) -> str:
        params = httpx.QueryParams({"q": query, "count": self.max_results})
        return f"https://www.bing.com/search?{params}"

    def _extract_cards(self, soup: BeautifulSoup, base_url: str) -> Sequence[_RawCard]:
        cards: List[_RawCard] = []
        for element in soup.select("li.b_algo"):
            anchor = element.find("a", href=True)
            if not anchor:
                continue
            title_node = element.find("h2")
            title = title_node.get_text(strip=True) if title_node else anchor.get_text(strip=True)
            snippet_node = element.select_one("div.b_caption p")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            cards.append(_RawCard(url=anchor["href"], title=title, snippet=snippet))
        return cards


class PlaywrightGoogleProvider(PlaywrightSerpProvider):
    ENGINE = "google"
    WAIT_FOR_SELECTOR = "div#search div.g div.yuRUbf > a"

    async def _after_navigation(self, page: Page) -> None:
        consent_selectors = [
            "button#L2AGLb",
            "button[aria-label='Tout accepter']",
            "button:has-text(\"Tout accepter\")",
            "button:has-text(\"Accept all\")",
        ]
        for selector in consent_selectors:
            try:
                button = await page.wait_for_selector(selector, timeout=2_000)
            except PlaywrightTimeoutError:
                continue
            if button is not None:
                try:
                    await button.click()
                    await page.wait_for_timeout(500)
                except PlaywrightTimeoutError:
                    pass
            break

    def _build_url(self, query: str) -> str:
        params = httpx.QueryParams({"q": query, "num": self.max_results, "hl": "fr"})
        return f"https://www.google.com/search?{params}"

    def _extract_cards(self, soup: BeautifulSoup, base_url: str) -> Sequence[_RawCard]:
        cards: List[_RawCard] = []
        for element in soup.select("div#search div.g"):
            anchor = element.select_one("div.yuRUbf > a[href]")
            if not anchor:
                continue
            title_node = anchor.select_one("h3")
            title = title_node.get_text(strip=True) if title_node else anchor.get_text(strip=True)
            if not title:
                continue
            snippet_node = element.select_one("div.VwiC3b, span.aCOpRe, div[data-sncf='1']")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            cards.append(_RawCard(url=anchor["href"], title=title, snippet=snippet))
        return cards


__all__ = ["PlaywrightSerpProvider", "PlaywrightBingProvider", "PlaywrightGoogleProvider"]
