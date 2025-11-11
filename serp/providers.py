from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Mapping
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse, unquote

import httpx
from bs4 import BeautifulSoup

from metrics.collector import get_metrics
from net.http_client import HttpClient
from utils.loggingx import get_logger

LOGGER = get_logger("serp.providers")
METRICS = get_metrics()

GENERIC_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "youtu.be",
    "viadeo.com",
    "pinterest.com",
    "wikipedia.org",
    "yelp.com",
    "tripadvisor.com",
    "trustpilot.com",
    "pagesjaunes.fr",
    "societe.com",
    "annuaire.com",
    "business.site",
}

TRACKING_PREFIXES = ("utm_", "utm", "icid", "mc_", "mkt_", "pk_", "spm")
TRACKING_KEYS = {
    "ref",
    "referrer",
    "gclid",
    "yclid",
    "msclkid",
    "fbclid",
    "igshid",
    "campaignid",
    "dclid",
}

ALLOWED_SCHEMES = {"http", "https"}


@dataclass(slots=True)
class Result:
    url: str
    domain: str
    title: str
    snippet: str
    rank: int


class SerpProvider:
    """Base class for SERP providers relying on HttpClient."""

    def __init__(self, cfg: Mapping[str, Any], http_client: HttpClient) -> None:
        self._cfg = dict(cfg)
        self._http = http_client
        self._max_results = max(1, int(self._cfg.get("max_results", 10)))

    @property
    def max_results(self) -> int:
        return self._max_results

    def search(self, query: str) -> List[Result]:
        raise NotImplementedError

    def _clean_url(self, raw_url: str, *, base: str | None = None) -> str:
        if not raw_url:
            return ""
        raw_url = raw_url.strip()
        if base and not urlparse(raw_url).scheme:
            raw_url = urljoin(base, raw_url)

        parsed = urlparse(raw_url)

        # DuckDuckGo HTML returns tracking redirectors in /l/?uddg=<url>.
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            for key, value in parse_qsl(parsed.query, keep_blank_values=True):
                if key == "uddg":
                    decoded = unquote(value)
                    if decoded and decoded != raw_url:
                        return self._clean_url(decoded)

        if parsed.scheme.lower() not in ALLOWED_SCHEMES or not parsed.netloc:
            return ""

        clean_params = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if not self._is_tracking_key(k)
        ]
        clean_query = urlencode(clean_params, doseq=True)

        cleaned = parsed._replace(query=clean_query, fragment="")
        return urlunparse(cleaned)

    def _extract_domain(self, url: str) -> str:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host

    def _is_generic_domain(self, domain: str) -> bool:
        return domain in GENERIC_DOMAINS

    def _is_tracking_key(self, key: str) -> bool:
        lowered = key.lower()
        if lowered in TRACKING_KEYS:
            return True
        return any(lowered.startswith(prefix) for prefix in TRACKING_PREFIXES)

class BingProvider(SerpProvider):
    """SERP provider parsing Bing HTML results."""

    def search(self, query: str) -> List[Result]:
        url = self._build_url(query)
        provider_name = type(self).__name__
        base_labels = {
            "provider": provider_name,
            "kind": "serp",
            "group": f"serp:{provider_name}",
        }
        METRICS.increment_counter("requests_total", labels=base_labels)
        start = time.perf_counter()
        response = self._http.get(url)
        duration = time.perf_counter() - start
        METRICS.record_latency(f"serp:{provider_name}", duration)
        if response.status_code != httpx.codes.OK:
            LOGGER.warning("Bing returned status %s for %r", response.status_code, query)
            error_labels = dict(base_labels)
            error_labels["status"] = str(response.status_code)
            METRICS.increment_counter("errors_total", labels=error_labels)
            return []
        html = response.text
        if not html:
            return []

        soup = BeautifulSoup(html, "lxml")

        # Bing SERP cards are located under <li class="b_algo"> with <h2><a> for titles and
        # a snippet within <div class="b_caption"><p>.
        items = soup.select("li.b_algo")
        results: List[Result] = []
        for element in items:
            anchor = element.find("a", href=True)
            if not anchor:
                continue
            cleaned_url = self._clean_url(anchor["href"], base=url)
            if not cleaned_url:
                continue
            domain = self._extract_domain(cleaned_url)
            if not domain or self._is_generic_domain(domain):
                continue

            title_node = element.find("h2")
            title = title_node.get_text(strip=True) if title_node else anchor.get_text(strip=True)
            snippet_node = element.select_one("div.b_caption p")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            results.append(
                Result(
                    url=cleaned_url,
                    domain=domain,
                    title=title,
                    snippet=snippet,
                    rank=len(results) + 1,
                )
            )
            if len(results) >= self.max_results:
                break
        return results

    def _build_url(self, query: str) -> str:
        params = httpx.QueryParams({"q": query, "count": self.max_results})
        return f"https://www.bing.com/search?{params}"


class DuckDuckGoProvider(SerpProvider):
    """SERP provider parsing DuckDuckGo Lite HTML results."""

    def search(self, query: str) -> List[Result]:
        url = self._build_url(query)
        provider_name = type(self).__name__
        base_labels = {
            "provider": provider_name,
            "kind": "serp",
            "group": f"serp:{provider_name}",
        }
        METRICS.increment_counter("requests_total", labels=base_labels)
        start = time.perf_counter()
        response = self._http.get(url)
        duration = time.perf_counter() - start
        METRICS.record_latency(f"serp:{provider_name}", duration)
        if response.status_code != httpx.codes.OK:
            LOGGER.warning("DuckDuckGo returned status %s for %r", response.status_code, query)
            error_labels = dict(base_labels)
            error_labels["status"] = str(response.status_code)
            METRICS.increment_counter("errors_total", labels=error_labels)
            return []
        html = response.text
        if not html:
            return []

        soup = BeautifulSoup(html, "lxml")

        # DuckDuckGo Lite exposes results in <div class="result">,
        # with <a class="result__a"> for titles and <a class="result__snippet"> for snippets.
        items = soup.select("div.result")
        results: List[Result] = []
        for element in items:
            anchor = element.select_one("a.result__a[href]")
            if not anchor:
                continue
            cleaned_url = self._clean_url(anchor["href"], base=url)
            if not cleaned_url:
                continue
            domain = self._extract_domain(cleaned_url)
            if not domain or self._is_generic_domain(domain):
                continue

            title = anchor.get_text(strip=True)
            snippet_node = element.select_one(".result__snippet, a.result__snippet")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            results.append(
                Result(
                    url=cleaned_url,
                    domain=domain,
                    title=title,
                    snippet=snippet,
                    rank=len(results) + 1,
                )
            )
            if len(results) >= self.max_results:
                break
        return results

    def _build_url(self, query: str) -> str:
        params = httpx.QueryParams({"q": query})
        return f"https://html.duckduckgo.com/html/?{params}"


__all__ = [
    "GENERIC_DOMAINS",
    "Result",
    "SerpProvider",
    "BingProvider",
    "DuckDuckGoProvider",
]
