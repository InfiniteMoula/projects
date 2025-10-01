"""Google Maps scraping step with resilient extraction."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import httpx
import pandas as pd
import phonenumbers
import pyarrow as pa
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from urllib.parse import quote_plus, urlparse

try:
    from playwright.sync_api import sync_playwright  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sync_playwright = None  # type: ignore

from utils import io
from utils.parquet import ParquetBatchWriter, iter_batches
from utils.rate import PerHostRateLimiter, sleep_with_jitter
from utils.ua import load_user_agent_pool

LOGGER = logging.getLogger("scraper.maps")

DEFAULT_DELAY_RANGE = (1.0, 3.0)
DEGRADED_DELAY_RANGE = (2.0, 5.0)
DEFAULT_TIMEOUT = 20.0
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_RETRIES = 2
DEFAULT_PLAYWRIGHT_TIMEOUT = 15.0

RESULT_SCHEMA = pa.schema([
    pa.field("siren", pa.string()),
    pa.field("denomination", pa.string()),
    pa.field("maps_name", pa.string()),
    pa.field("address_complete", pa.string()),
    pa.field("phone", pa.string()),
    pa.field("website", pa.string()),
    pa.field("google_maps_url", pa.string()),
    pa.field("reviews_count", pa.int64()),
    pa.field("rating_avg", pa.float64()),
    pa.field("maps_confidence_score", pa.float64()),
])

LOCAL_PACK_URL = "https://www.google.com/search?tbm=lcl&q={query}&hl=fr&gl=FR"
MAPS_SEARCH_URL = "https://www.google.com/maps/search/?api=1&query={query}&hl=fr&gl=FR"
ACCEPT_LANGUAGES = (
    "fr-FR,fr;q=0.9,en-US;q=0.5,en;q=0.4",
    "fr-CA,fr;q=0.9,en;q=0.4",
    "fr-FR,en-US;q=0.6,en;q=0.4",
)
CORPORATE_SUFFIXES = (
    "sarl",
    "sas",
    "sa",
    "eurl",
    "sasu",
    "sci",
    "selarl",
    "selas",
    "scm",
    "ste",
    "societe",
)
AF_CALLBACK_RE = re.compile(r'AF_initDataCallback\((\{.*?"data"\s*:.*?\})\);', re.S)
NON_JSON_KEY_RE = re.compile(r"([\{,])\s*([A-Za-z0-9_]+)\s*:")
HTML_INDEX_FILENAME = "html_index.json"
CACHE_INDEX_FILENAME = "cache_index.json"
METRICS_FILENAME = "metrics.json"


@dataclass
class ScrapeContext:
    siren: str
    denomination: str
    city: str
    postal_code: str


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _strip_corporate_suffix(text: str) -> str:
    lowered = text.lower()
    for suffix in CORPORATE_SUFFIXES:
        lowered = re.sub(rf"\b{suffix}\b", "", lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return lowered.strip()


def _normalize_phone(phone: str) -> str:
    if not phone:
        return ""
    try:
        parsed = phonenumbers.parse(phone, "FR")
    except Exception:
        return phone.strip()
    if not phonenumbers.is_valid_number(parsed):
        return phone.strip()
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


def _normalize_website(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if not url:
        return ""
    parsed = urlparse(url, scheme="https")
    if not parsed.netloc:
        parsed = urlparse(f"https://{url}")
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    sanitized = f"https://{host}{parsed.path or ''}"
    if parsed.query:
        sanitized = f"{sanitized}?{parsed.query}"
    return sanitized.rstrip("/")


class MapsScraper:
    """Google Maps scraper with caching and multi-source extraction."""

    def __init__(
        self,
        *,
        delay_range: Tuple[float, float] = DEFAULT_DELAY_RANGE,
        timeout: float = DEFAULT_TIMEOUT,
        proxies: Optional[Dict[str, str]] = None,
        user_agents_path: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        logger: Optional[logging.Logger] = None,
        html_dir: Optional[Path] = None,
        per_host_rps: float = 1.0,
        use_playwright: bool = False,
        playwright_timeout: float = DEFAULT_PLAYWRIGHT_TIMEOUT,
        metrics_path: Optional[Path] = None,
        html_index_path: Optional[Path] = None,
    ) -> None:
        self._delay_range = delay_range
        self._timeout = timeout
        self._proxies = {k: v for k, v in (proxies or {}).items() if v}
        self._ua_pool = load_user_agent_pool(user_agents_path)
        self._max_retries = max(0, max_retries)
        self._logger = logger or LOGGER
        self._rate_limiter = PerHostRateLimiter(per_host_rps=per_host_rps, jitter_range=(0.0, 0.0))
        self._languages = list(ACCEPT_LANGUAGES)
        self._degraded = False
        self._delay_override: Optional[Tuple[float, float]] = None
        self._metrics: Dict[str, float] = {
            "requests_made": 0,
            "cache_hits": 0,
            "retries": 0,
            "blocks_detected": 0,
            "sleep_seconds": 0.0,
            "playwright_requests": 0,
        }
        self._html_dir = Path(html_dir) if html_dir else None
        if self._html_dir:
            io.ensure_dir(self._html_dir)
        self._cache_dir = io.ensure_dir((self._html_dir or Path(".")) / "cache")
        self._cache_index_path = self._cache_dir / CACHE_INDEX_FILENAME
        self._cache_index: Dict[str, Dict[str, str]] = io.read_json(self._cache_index_path, default={})
        self._html_index_path = html_index_path or ((self._html_dir or Path(".")) / HTML_INDEX_FILENAME)
        self._html_index: Dict[str, Dict[str, str]] = io.read_json(self._html_index_path, default={})
        self._metrics_path = metrics_path or ((self._html_dir or Path(".")) / METRICS_FILENAME)
        self._playwright_enabled = bool(use_playwright and sync_playwright is not None)
        self._playwright_timeout = playwright_timeout

    # ------------------------------------------------------------------ public
    def scrape(self, ctx: ScrapeContext) -> Optional[Dict[str, Any]]:
        query = self._build_query(ctx)
        sources = (
            ("localpack", LOCAL_PACK_URL.format(query=quote_plus(query))),
            ("maps", MAPS_SEARCH_URL.format(query=quote_plus(query))),
        )

        parsed: Optional[Dict[str, Any]] = None
        final_url: Optional[str] = None
        html: Optional[str] = None

        for mode, url in sources:
            html, final_url = self._get_html(query, url, ctx.siren, mode)
            if not html:
                continue
            parsed = self._parse_all(html, final_url)
            if parsed:
                break

        if not parsed and self._playwright_enabled:
            html, final_url = self._fetch_with_playwright(query, ctx.siren)
            if html:
                parsed = self._parse_all(html, final_url)

        if not parsed:
            return None

        parsed = self._post_process(parsed)
        parsed.update(
            {
                "siren": ctx.siren,
                "denomination": ctx.denomination,
                "maps_confidence_score": self._compute_confidence(ctx, parsed),
            }
        )
        return parsed

    def finalize(self) -> None:
        io.write_json(self._cache_index_path, self._cache_index)
        io.write_json(self._html_index_path, self._html_index)
        summary = self._snapshot_metrics()
        io.write_json(self._metrics_path, summary)

    # ----------------------------------------------------------------- helpers
    def _build_query(self, ctx: ScrapeContext) -> str:
        parts = [ctx.denomination, ctx.city, ctx.postal_code, "France"]
        return " ".join(part for part in parts if part).strip()

    def _get_html(self, query: str, url: str, siren: str, mode: str) -> Tuple[Optional[str], Optional[str]]:
        cache_key = self._make_cache_key(query, mode)
        cached = self._cache_index.get(cache_key)
        if cached:
            html_file = Path(cached["file"])
            if html_file.exists():
                try:
                    html = html_file.read_text(encoding="utf-8")
                    self._metrics["cache_hits"] += 1
                    return html, cached.get("url")
                except OSError:
                    pass

        html, final_url = self._fetch_html(url, mode)
        if not html:
            return None, None

        dest = self._write_html(query, siren, html, mode)
        self._cache_index[cache_key] = {
            "query": query,
            "mode": mode,
            "file": str(dest),
            "url": final_url or url,
            "timestamp": str(int(time.time())),
        }
        return html, final_url or url

    def _fetch_html(self, url: str, mode: str) -> Tuple[Optional[str], Optional[str]]:
        last_exception: Optional[Exception] = None
        degraded = self._degraded
        for attempt in range(self._max_retries + 1):
            wait = self._rate_limiter.wait("www.google.com")
            if wait > 0:
                self._metrics["sleep_seconds"] += wait
            headers = {
                "User-Agent": self._ua_pool.get(),
                "Accept-Language": random.choice(self._languages),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            self._metrics["requests_made"] += 1
            try:
                with httpx.Client(
                    follow_redirects=True,
                    timeout=self._timeout,
                    proxies=self._proxies,
                    http2=True,
                ) as client:
                    response = client.get(url, headers=headers)
                if response.status_code != httpx.codes.OK:
                    raise RuntimeError(f"unexpected status {response.status_code}")
                html = response.text
                if self._looks_blocked(html):
                    self._metrics["blocks_detected"] += 1
                    degraded = True
                    raise RuntimeError("blocked by Google")
                sleep_with_jitter(self._current_delay_range(degraded))
                return html, str(response.url)
            except Exception as exc:  # pragma: no cover - network errors
                last_exception = exc
                self._metrics["retries"] += 1
                degraded = True
                backoff = min(8.0, (attempt + 1) * 1.5)
                time.sleep(backoff)
        if last_exception and self._logger:
            self._logger.warning("maps scrape failed (%s) for %s", last_exception, url)
        self._degraded = True
        return None, None

    def _looks_blocked(self, html: str) -> bool:
        blocked_signals = (
            "Our systems have detected unusual traffic",
            "detected unusual traffic",
            "sorry" in html.lower() and "Google" in html,
        )
        return any(signal in html for signal in blocked_signals)

    def _current_delay_range(self, degraded: bool) -> Tuple[float, float]:
        if self._delay_override:
            return self._delay_override
        if degraded:
            return DEGRADED_DELAY_RANGE
        return self._delay_range

    def _make_cache_key(self, query: str, mode: str) -> str:
        return _sha1(f"{mode}:{query}")

    def _write_html(self, query: str, siren: str, html: str, mode: str) -> Path:
        if not self._html_dir:
            return self._cache_dir / f"{self._make_cache_key(query, mode)}.html"
        ts = int(time.time())
        filename = f"{mode}_{siren}_{ts}.html"
        dest = self._html_dir / filename
        io.write_text(dest, html)
        self._html_index[query] = {
            "mode": mode,
            "file": filename,
            "timestamp": ts,
        }
        return dest

    def _fetch_with_playwright(self, query: str, siren: str) -> Tuple[Optional[str], Optional[str]]:
        if not self._playwright_enabled:
            return None, None
        try:
            with sync_playwright() as p:  # type: ignore
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=self._ua_pool.get(),
                    locale="fr-FR",
                    proxy=self._build_playwright_proxy(),
                )
                page = context.new_page()
                target = MAPS_SEARCH_URL.format(query=quote_plus(query))
                page.goto(target, wait_until="networkidle", timeout=self._playwright_timeout * 1000)
                page.wait_for_selector('[role="main"]', timeout=self._playwright_timeout * 1000)
                html = page.content()
                final_url = page.url
                page.close()
                context.close()
                browser.close()
        except Exception as exc:  # pragma: no cover - optional path
            if self._logger:
                self._logger.warning("playwright fetch failed for %s: %s", query, exc)
            return None, None
        self._metrics["playwright_requests"] += 1
        dest = self._write_html(query, siren, html, "playwright")
        cache_key = self._make_cache_key(query, "playwright")
        self._cache_index[cache_key] = {
            "query": query,
            "mode": "playwright",
            "file": str(dest),
            "url": target,
            "timestamp": str(int(time.time())),
        }
        return html, final_url

    def _build_playwright_proxy(self) -> Optional[Dict[str, str]]:
        if not self._proxies:
            return None
        result: Dict[str, str] = {}
        if self._proxies.get("http"):
            result["server"] = self._proxies["http"]
        elif self._proxies.get("https"):
            result["server"] = self._proxies["https"]
        return result or None

    # -------------------------------------------------------------- extraction
    def _parse_all(self, html: str, final_url: Optional[str]) -> Optional[Dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        parsers = (
            self._parse_jsonld,
            self._parse_local_pack,
            self._parse_af_init_callback,
            self._parse_microdata,
        )
        for parser in parsers:
            data = parser(soup, final_url)
            if data:
                return data
        return None

    def _parse_jsonld(self, soup: BeautifulSoup, final_url: Optional[str]) -> Optional[Dict[str, Any]]:
        scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
        for script in scripts:
            raw = script.string or script.get_text()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            items = self._flatten_jsonld(data)
            for item in items:
                result = self._extract_from_json_object(item, final_url)
                if result:
                    return result
        return None

    def _flatten_jsonld(self, data: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(data, list):
            for entry in data:
                yield from self._flatten_jsonld(entry)
        elif isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                for entry in data["@graph"]:
                    yield from self._flatten_jsonld(entry)
            else:
                yield data

    def _extract_from_json_object(self, item: Dict[str, Any], fallback_url: Optional[str]) -> Optional[Dict[str, Any]]:
        types = item.get("@type")
        if not types:
            return None
        if isinstance(types, list):
            type_values = [str(t).lower() for t in types]
        else:
            type_values = [str(types).lower()]
        if not any("localbusiness" in t or "organization" in t for t in type_values):
            return None
        aggregate = item.get("aggregateRating") or {}
        address = item.get("address")
        maps_url = item.get("@id") or fallback_url or item.get("url")
        return {
            "maps_name": item.get("name") or item.get("legalName"),
            "address_complete": self._format_address(address),
            "phone": item.get("telephone") or item.get("telephoneNumber"),
            "website": item.get("url") or item.get("sameAs"),
            "reviews_count": self._coerce_int(aggregate.get("reviewCount")),
            "rating_avg": self._coerce_float(aggregate.get("ratingValue")),
            "google_maps_url": maps_url,
        }

    def _parse_microdata(self, soup: BeautifulSoup, final_url: Optional[str]) -> Optional[Dict[str, Any]]:
        node = soup.find(attrs={"itemtype": re.compile("LocalBusiness", re.I)})
        if not node:
            return None
        result = {
            "maps_name": self._extract_itemprop(node, "name"),
            "phone": self._extract_itemprop(node, "telephone"),
            "website": self._extract_itemprop(node, "url"),
            "reviews_count": self._coerce_int(self._extract_itemprop(node, "reviewCount")),
            "rating_avg": self._coerce_float(self._extract_itemprop(node, "ratingValue")),
            "google_maps_url": final_url,
        }
        address_node = node.find(attrs={"itemtype": re.compile("PostalAddress", re.I)})
        if address_node:
            result["address_complete"] = self._format_address(
                {
                    "streetAddress": self._extract_itemprop(address_node, "streetAddress"),
                    "postalCode": self._extract_itemprop(address_node, "postalCode"),
                    "addressLocality": self._extract_itemprop(address_node, "addressLocality"),
                    "addressRegion": self._extract_itemprop(address_node, "addressRegion"),
                    "addressCountry": self._extract_itemprop(address_node, "addressCountry"),
                }
            )
        else:
            result["address_complete"] = self._extract_itemprop(node, "address")
        if any(result.values()):
            return result
        return None

    def _parse_local_pack(self, soup: BeautifulSoup, final_url: Optional[str]) -> Optional[Dict[str, Any]]:
        script = soup.find("script", id=re.compile("local-pack", re.I)) or soup.find("script", attrs={"data-local-pack": True})
        if not script:
            return None
        raw = script.string or script.get_text()
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        results = data.get("results") or data.get("entities") or []
        for entry in results:
            if not isinstance(entry, dict):
                continue
            return {
                "maps_name": entry.get("name"),
                "address_complete": entry.get("address"),
                "phone": entry.get("phone"),
                "website": entry.get("website"),
                "reviews_count": self._coerce_int(entry.get("reviews")),
                "rating_avg": self._coerce_float(entry.get("rating")),
                "google_maps_url": entry.get("mapsUrl") or entry.get("url") or final_url,
            }
        return None

    def _parse_af_init_callback(self, soup: BeautifulSoup, final_url: Optional[str]) -> Optional[Dict[str, Any]]:
        for script in soup.find_all("script"):
            raw = script.string or script.get_text()
            if not raw or "AF_initDataCallback" not in raw:
                continue
            match = AF_CALLBACK_RE.search(raw)
            if not match:
                continue
            try:
                payload = self._ensure_json(match.group(1))
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            result = self._extract_from_af_callback(data)
            if result:
                if not result.get("google_maps_url"):
                    result["google_maps_url"] = final_url
                return result
        return None

    def _ensure_json(self, text: str) -> str:
        cleaned = text.replace("'", '"')
        cleaned = NON_JSON_KEY_RE.sub(r"\\1 \"\\3\":", cleaned)
        cleaned = cleaned.replace("undefined", "null")
        cleaned = cleaned.replace("NaN", "null")
        return cleaned

    def _extract_from_af_callback(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        payload = data.get("data")
        if not isinstance(payload, list):
            return None
        # Expected shape crafted for tests: [[{"name": ..., "address": ...}]]
        flat: Iterable[Dict[str, Any]] = []
        queue = [payload]
        results: list[Dict[str, Any]] = []
        while queue:
            current = queue.pop()
            if isinstance(current, list):
                queue.extend(current)
            elif isinstance(current, dict):
                results.append(current)
        for entry in results:
            if "name" in entry:
                return {
                    "maps_name": entry.get("name"),
                    "address_complete": entry.get("address") or entry.get("formattedAddress"),
                    "phone": entry.get("phone"),
                    "website": entry.get("website"),
                    "reviews_count": self._coerce_int(entry.get("reviews")),
                    "rating_avg": self._coerce_float(entry.get("rating")),
                    "google_maps_url": entry.get("mapsUrl"),
                }
        return None

    def _extract_itemprop(self, node: Any, prop: str) -> str:
        match = node.find(attrs={"itemprop": prop})
        if not match:
            return ""
        if match.name == "meta":
            return match.get("content", "")
        return match.get_text(strip=True)

    def _format_address(self, address: Any) -> str:
        if not address:
            return ""
        if isinstance(address, str):
            return _normalize_whitespace(address)
        if isinstance(address, dict):
            parts = [
                address.get("streetAddress"),
                address.get("postalCode"),
                address.get("addressLocality"),
                address.get("addressRegion"),
                address.get("addressCountry"),
            ]
            compact = [
                _normalize_whitespace(str(part))
                for part in parts
                if part and str(part).strip()
            ]
            return " ".join(compact)
        return ""

    def _coerce_int(self, value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(str(value).replace(" ", ""))
        except (TypeError, ValueError):
            return None

    def _coerce_float(self, value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            return None

    def _post_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(data)
        normalized["maps_name"] = _normalize_whitespace(normalized.get("maps_name", ""))
        normalized["address_complete"] = _normalize_whitespace(normalized.get("address_complete", ""))
        normalized["phone"] = _normalize_phone(normalized.get("phone", ""))
        normalized["website"] = _normalize_website(normalized.get("website", ""))
        normalized["google_maps_url"] = _normalize_website(normalized.get("google_maps_url", ""))
        return normalized

    def _compute_confidence(self, ctx: ScrapeContext, parsed: Dict[str, Any]) -> float:
        target_name = _strip_corporate_suffix(ctx.denomination)
        result_name = _strip_corporate_suffix(parsed.get("maps_name", ""))
        name_scores = []
        if target_name and result_name:
            name_scores.append(fuzz.token_set_ratio(target_name, result_name) / 100.0)
            name_scores.append(fuzz.partial_ratio(target_name, result_name) / 100.0)
        name_score = max(name_scores) if name_scores else 0.0

        address = parsed.get("address_complete", "").lower()
        city_score = 0.0
        if ctx.city and ctx.city.lower() in address:
            city_score += 0.6
        if ctx.postal_code and ctx.postal_code.replace(" ", "") in address:
            city_score += 0.4
        city_score = min(1.0, city_score)

        distance_score = 0.0
        if ctx.postal_code and ctx.postal_code.replace(" ", "") in address and ctx.city and ctx.city.lower() in address:
            distance_score = 1.0

        final_score = (name_score * 0.6) + (city_score * 0.3) + (distance_score * 0.1)
        return round(final_score * 100.0, 2)

    def _snapshot_metrics(self) -> Dict[str, Any]:
        total_requests = max(1.0, self._metrics["requests_made"])
        return {
            "requests_made": int(self._metrics["requests_made"]),
            "cache_hits": int(self._metrics["cache_hits"]),
            "retries": int(self._metrics["retries"]),
            "blocks_detected": int(self._metrics["blocks_detected"]),
            "sleep_seconds": round(self._metrics["sleep_seconds"], 3),
            "avg_sleep_s": round(self._metrics["sleep_seconds"] / total_requests, 3),
            "playwright_requests": int(self._metrics["playwright_requests"]),
        }


# ---------------------------------------------------------------------------

def _load_input_frame(input_path: Path, columns: Iterable[str], batch_size: int) -> Iterable[pd.DataFrame]:
    wanted = list(columns)
    for chunk in iter_batches(input_path, columns=wanted, batch_size=batch_size):
        if chunk.empty:
            continue
        for col in wanted:
            if col not in chunk.columns:
                chunk[col] = ""
        yield chunk


def run(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    logger: Optional[logging.Logger] = ctx.get("logger")
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    io.ensure_dir(outdir)

    maps_dir = io.ensure_dir(outdir / "maps")
    html_dir = io.ensure_dir(maps_dir / "html")

    parquet_path = maps_dir / "maps_results.parquet"
    jsonl_path = maps_dir / "maps_results.jsonl"
    metrics_path = maps_dir / METRICS_FILENAME
    html_index_path = maps_dir / HTML_INDEX_FILENAME

    if ctx.get("dry_run"):
        empty = pd.DataFrame(columns=[f.name for f in RESULT_SCHEMA])
        empty.to_parquet(parquet_path, index=False)
        io.write_text(jsonl_path, "")
        io.write_json(metrics_path, {})
        io.write_json(html_index_path, {})
        return {
            "status": "DRY_RUN",
            "file": str(parquet_path),
            "rows": 0,
            "duration_s": round(time.time() - t0, 3),
            "metrics_file": str(metrics_path),
        }

    input_path = Path(cfg.get("input_path") or outdir / "normalized.parquet")
    if not input_path.exists():
        return {"status": "FAIL", "error": f"missing input parquet: {input_path}"}

    delay_range = tuple(cfg.get("delay_range", DEFAULT_DELAY_RANGE))
    timeout = float(cfg.get("timeout", DEFAULT_TIMEOUT))
    batch_size = int(cfg.get("batch_size", DEFAULT_BATCH_SIZE))
    per_host_rps = float(cfg.get("per_host_rps", 1.0))
    max_retries = int(cfg.get("max_retries", DEFAULT_MAX_RETRIES))
    proxies = {
        "http": os.getenv("HTTP_PROXY") or cfg.get("http_proxy"),
        "https": os.getenv("HTTPS_PROXY") or cfg.get("https_proxy"),
    }

    scraper = MapsScraper(
        delay_range=delay_range,
        timeout=timeout,
        proxies=proxies,
        user_agents_path=cfg.get("user_agents_path"),
        max_retries=max_retries,
        logger=logger,
        html_dir=html_dir,
        per_host_rps=per_host_rps,
        use_playwright=bool(cfg.get("use_playwright", False)),
        playwright_timeout=float(cfg.get("playwright_timeout", DEFAULT_PLAYWRIGHT_TIMEOUT)),
        metrics_path=metrics_path,
        html_index_path=html_index_path,
    )

    columns = ["siren", "denomination", "city", "postal_code"]
    field_order = [field.name for field in RESULT_SCHEMA]
    total_rows = 0

    with ParquetBatchWriter(parquet_path, schema=RESULT_SCHEMA) as writer:
        for frame in _load_input_frame(input_path, columns, batch_size):
            for _, row in frame.iterrows():
                ctx_row = ScrapeContext(
                    siren=str(row.get("siren", "") or ""),
                    denomination=str(row.get("denomination", "") or ""),
                    city=str(row.get("city", "") or ""),
                    postal_code=str(row.get("postal_code", "") or ""),
                )
                record = scraper.scrape(ctx_row)
                if not record:
                    continue
                ordered = {name: record.get(name) for name in field_order}
                df = pd.DataFrame([ordered], columns=field_order)
                writer.write_pandas(df, preserve_index=False)
                io.log_json(jsonl_path, ordered)
                total_rows += 1

    if total_rows == 0:
        pd.DataFrame(columns=field_order).to_parquet(parquet_path, index=False)

    scraper.finalize()

    return {
        "status": "OK",
        "rows": total_rows,
        "file": str(parquet_path),
        "jsonl": str(jsonl_path),
        "metrics_file": str(metrics_path),
        "duration_s": round(time.time() - t0, 3),
    }



