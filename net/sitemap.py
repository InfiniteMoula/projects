from __future__ import annotations

import logging
from typing import List, Set
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

from net.http_client import HttpClient
from net import robots

LOGGER = logging.getLogger("net.sitemap")


def discover_sitemap_urls(site_url: str, http_client: HttpClient) -> List[str]:
    """
    Discover candidate page URLs from sitemaps for *site_url*.

    The function looks for ``/sitemap.xml`` and any ``Sitemap:`` entries declared in
    ``robots.txt``. Extracted URLs are limited to entries ending with ``/contact`` or
    containing ``mentions`` to reduce the total volume.
    """
    if not site_url:
        return []

    robots.configure(http_client)

    user_agent = _resolve_user_agent(http_client)
    parsed_site = urlparse(site_url)
    if not parsed_site.scheme:
        parsed_site = parsed_site._replace(scheme="https")
    if not parsed_site.netloc:
        return []
    site_root = f"{parsed_site.scheme}://{parsed_site.netloc}"

    sitemap_urls: List[str] = []
    seen_sitemaps: Set[str] = set()

    def add_sitemap(target: str) -> None:
        candidate = (target or "").strip()
        if not candidate:
            return
        if candidate not in seen_sitemaps:
            seen_sitemaps.add(candidate)
            sitemap_urls.append(candidate)

    add_sitemap(urljoin(site_root + "/", "sitemap.xml"))

    try:
        parser = robots.get_parser(site_root, user_agent)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("Unable to load robots parser for %s: %s", site_root, exc)
        parser = None
    if parser:
        for entry in parser.site_maps() or []:
            add_sitemap(entry)

    discovered: List[str] = []
    seen_pages: Set[str] = set()
    for sitemap_url in sitemap_urls:
        try:
            response = http_client.get(sitemap_url)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Sitemap fetch failed for %s: %s", sitemap_url, exc)
            continue
        if response.status_code != 200 or not response.text:
            continue
        try:
            root = ElementTree.fromstring(response.text)
        except ElementTree.ParseError as exc:
            LOGGER.debug("Invalid sitemap XML for %s: %s", sitemap_url, exc)
            continue
        for loc in root.iter("{*}loc"):
            if not loc.text:
                continue
            href = loc.text.strip()
            if not href:
                continue
            parsed_href = urlparse(href)
            if parsed_href.netloc and parsed_href.netloc.lower() != parsed_site.netloc.lower():
                continue
            lower = href.lower()
            if lower.endswith("/contact") or "mention" in lower:
                if href not in seen_pages:
                    seen_pages.add(href)
                    discovered.append(href)
    return discovered


def _resolve_user_agent(http_client: HttpClient) -> str:
    try:
        picker = getattr(http_client, "pick_user_agent")
    except AttributeError:  # pragma: no cover - defensive
        return "Mozilla/5.0 (compatible; SitemapBot/1.0; +https://example.com/bot)"
    try:
        return picker()
    except Exception:  # pragma: no cover - defensive
        return "Mozilla/5.0 (compatible; SitemapBot/1.0; +https://example.com/bot)"


__all__ = ["discover_sitemap_urls"]
