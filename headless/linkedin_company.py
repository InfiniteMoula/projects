from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Dict, Mapping, Optional

try:
    from playwright.async_api import (
        Error as PlaywrightError,
        Page,
        TimeoutError as PlaywrightTimeoutError,
        async_playwright,
    )
except ImportError:  # pragma: no cover - optional dependency
    async_playwright = None  # type: ignore
    PlaywrightError = Exception  # type: ignore[misc,assignment]
    PlaywrightTimeoutError = Exception  # type: ignore[misc,assignment]
    Page = Any  # type: ignore[misc,assignment]


LOGGER = logging.getLogger("headless.linkedin_company")

DEFAULT_TIMEOUT_S = 20.0
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
_WHITESPACE_RE = re.compile(r"\s+")

_COOKIE_ACCEPT_SELECTORS = (
    "button[data-tracking-control-name*='cookie.consent.accept']",
    "button[data-control-name*='cookie.consent.accept']",
    "button[aria-label*='Accept']",
    "button[aria-label*='Accepter']",
    "button:has-text('Accept')",
    "button:has-text('Accepter')",
    "button:has-text('Aceptar')",
    "button:has-text('Agree')",
)

_ABOUT_SELECTORS = {
    "sector": "div[data-test-id='about-us__industry'] dd",
    "employee_range": "div[data-test-id='about-us__size'] dd",
    "description": "[data-test-id='about-us__description']",
    "location": "div[data-test-id='about-us__headquarters'] dd",
}


def fetch_linkedin_company_profile(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    headless: bool = True,
    locale: str = "fr-FR",
) -> Dict[str, Optional[str]]:
    """
    Synchronous helper that scrapes a public LinkedIn company page with Playwright.
    """
    if async_playwright is None:  # pragma: no cover - runtime safeguard
        raise RuntimeError("Playwright is not installed. Please install playwright to use this function.")
    return _run_sync(_scrape_linkedin_company(url, timeout=timeout, headless=headless, locale=locale))


async def _scrape_linkedin_company(
    url: str,
    *,
    timeout: float,
    headless: bool,
    locale: str,
) -> Dict[str, Optional[str]]:
    if async_playwright is None:  # pragma: no cover - runtime safeguard
        raise RuntimeError("Playwright is not installed. Please install playwright to use this function.")

    playwright = await async_playwright().start()
    browser = None
    context = None
    page: Optional[Page] = None
    timeout_ms = max(1_000, int(timeout * 1_000))

    try:
        browser = await playwright.chromium.launch(
            headless=headless,
            args=(
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ),
        )
        context = await browser.new_context(
            user_agent=DEFAULT_USER_AGENT,
            locale=locale,
            java_script_enabled=True,
        )
        context.set_default_navigation_timeout(timeout_ms)
        page = await context.new_page()
        await page.route("**/*", _abort_noise)

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(f"Timed out while loading LinkedIn page: {url}") from exc

        await _try_accept_cookies(page)
        try:
            await page.wait_for_selector("h1.top-card-layout__title", timeout=timeout_ms // 2)
        except PlaywrightTimeoutError:
            LOGGER.debug("LinkedIn company name selector not found: %s", url)

        await page.wait_for_timeout(500)

        data = await _extract_company_data(page)
        if not data.get("logo_url"):
            data["logo_url"] = await _extract_logo_fallback(page)
        return data
    finally:
        if page is not None:
            try:
                await page.close()
            except PlaywrightError:  # pragma: no cover - defensive
                pass
        if context is not None:
            try:
                await context.close()
            except PlaywrightError:  # pragma: no cover - defensive
                pass
        if browser is not None:
            try:
                await browser.close()
            except PlaywrightError:  # pragma: no cover - defensive
                pass
    if playwright is not None:
        try:
            await playwright.stop()
        except PlaywrightError:  # pragma: no cover - defensive
            pass


async def _extract_company_data(page: Page) -> Dict[str, Optional[str]]:
    json_ld = await _extract_json_ld(page)

    name = await _get_text(page, "h1.top-card-layout__title")
    sector = await _get_text(page, _ABOUT_SELECTORS["sector"])
    employee_range = await _get_text(page, _ABOUT_SELECTORS["employee_range"])
    description = await _get_text(page, _ABOUT_SELECTORS["description"])
    location = await _get_text(page, _ABOUT_SELECTORS["location"])
    logo_url = await _get_logo_from_page(page)

    if not name:
        name = _clean_text(json_ld.get("name"))

    if not description:
        description = _clean_text(json_ld.get("description"))

    if not location:
        location = _format_address(json_ld.get("address"))

    if not logo_url and isinstance(json_ld.get("logo"), Mapping):
        logo_url = _clean_text(json_ld["logo"].get("contentUrl"))

    if not employee_range:
        employee_range = _format_employee_count(json_ld.get("numberOfEmployees"))

    return {
        "name": name,
        "sector": sector,
        "employee_range": employee_range,
        "description": description,
        "location": location,
        "logo_url": logo_url,
    }


async def _get_text(page: Page, selector: str) -> Optional[str]:
    locator = page.locator(selector)
    try:
        element = locator.first
        text = await element.inner_text(timeout=3_000)
    except PlaywrightTimeoutError:
        return None
    except PlaywrightError:  # pragma: no cover - defensive
        return None
    return _clean_text(text)


async def _get_logo_from_page(page: Page) -> Optional[str]:
    try:
        logo = await page.evaluate(
            """() => {
                const candidate = document.querySelector('img.top-card-layout__entity-image, img[data-test-id="hero-company-logo"]');
                if (!candidate) {
                    return null;
                }
                return candidate.getAttribute('src')
                    || candidate.getAttribute('data-delayed-url')
                    || candidate.getAttribute('data-ghost-url');
            }"""
        )
    except PlaywrightError:  # pragma: no cover - defensive
        logo = None
    return _clean_text(logo)


async def _extract_logo_fallback(page: Page) -> Optional[str]:
    try:
        meta_logo = await page.locator("meta[property='og:image']").first.get_attribute("content")
    except PlaywrightTimeoutError:
        meta_logo = None
    except PlaywrightError:  # pragma: no cover - defensive
        meta_logo = None
    return _clean_text(meta_logo)


async def _extract_json_ld(page: Page) -> Dict[str, Any]:
    scripts = page.locator("script[type='application/ld+json']")
    data: Dict[str, Any] = {}
    count = await scripts.count()
    for idx in range(count):
        try:
            raw = await scripts.nth(idx).inner_text()
        except PlaywrightError:  # pragma: no cover - defensive
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        candidate = _find_organization_node(parsed)
        if candidate:
            data.update(candidate)
            break
    return data


def _find_organization_node(parsed: Any) -> Optional[Dict[str, Any]]:
    if isinstance(parsed, Mapping):
        if parsed.get("@type") == "Organization":
            return dict(parsed)
        graph = parsed.get("@graph")
        if graph:
            return _find_organization_node(graph)
    if isinstance(parsed, list):
        for item in parsed:
            result = _find_organization_node(item)
            if result:
                return result
    return None


async def _try_accept_cookies(page: Page) -> None:
    for selector in _COOKIE_ACCEPT_SELECTORS:
        try:
            button = page.locator(selector).first
            await button.click(timeout=2_000)
            await page.wait_for_timeout(400)
            return
        except PlaywrightTimeoutError:
            continue
        except PlaywrightError:  # pragma: no cover - defensive
            continue


async def _abort_noise(route, request) -> None:
    resource_type = request.resource_type
    if resource_type in {"image", "font", "stylesheet", "media"}:
        await route.abort()
    else:
        await route.continue_()


def _format_address(address: Any) -> Optional[str]:
    if not isinstance(address, Mapping):
        return None
    parts = [
        _clean_text(address.get("addressLocality")),
        _clean_text(address.get("addressRegion")),
        _clean_text(address.get("addressCountry")),
    ]
    parts = [part for part in parts if part]
    if not parts:
        street = _clean_text(address.get("streetAddress"))
        return street
    return ", ".join(dict.fromkeys(parts))


def _format_employee_count(number: Any) -> Optional[str]:
    if isinstance(number, Mapping):
        number = number.get("value")
    if number is None:
        return None
    try:
        value = int(number)
    except (TypeError, ValueError):
        return _clean_text(number)
    if value <= 1:
        return "1 employee"
    if value < 10:
        return f"{value} employees"
    if value < 50:
        return "11-50 employees"
    if value < 200:
        return "51-200 employees"
    if value < 500:
        return "201-500 employees"
    if value < 1000:
        return "501-1,000 employees"
    if value < 5000:
        return "1,001-5,000 employees"
    if value < 10000:
        return "5,001-10,000 employees"
    return "10,001+ employees"


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = _WHITESPACE_RE.sub(" ", value).strip()
    return cleaned or None


def _run_sync(coro: Awaitable[Any]) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


__all__ = ["fetch_linkedin_company_profile"]
