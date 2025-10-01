"""Google Maps scraping step."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import httpx
import pandas as pd
import pyarrow as pa
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from urllib.parse import quote_plus

from utils import io
from utils.parquet import ParquetBatchWriter, iter_batches
from utils.rate import PerHostRateLimiter, sleep_with_jitter
from utils.ua import load_user_agent_pool

LOGGER = logging.getLogger("scraper.maps")

DEFAULT_DELAY_RANGE = (1.0, 3.0)
DEFAULT_TIMEOUT = 15.0
DEFAULT_BATCH_SIZE = 128
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


@dataclass
class ScrapeContext:
    siren: str
    denomination: str
    city: str
    postal_code: str


class MapsScraper:
    """Lightweight Google Maps scraper with respect for rate limits."""

    def __init__(
        self,
        *,
        delay_range: Tuple[float, float] = DEFAULT_DELAY_RANGE,
        timeout: float = DEFAULT_TIMEOUT,
        proxies: Optional[Dict[str, str]] = None,
        user_agents_path: Optional[str] = None,
        max_retries: int = 2,
        logger: Optional[logging.Logger] = None,
        html_dir: Optional[Path] = None,
        per_host_rps: float = 1.0,
    ) -> None:
        self._delay_range = delay_range
        self._timeout = timeout
        self._proxies = {k: v for k, v in (proxies or {}).items() if v}
        self._ua_pool = load_user_agent_pool(user_agents_path)
        self._max_retries = max(0, int(max_retries))
        self._logger = logger or LOGGER
        self._html_dir = Path(html_dir) if html_dir else None
        if self._html_dir:
            io.ensure_dir(self._html_dir)
        self._rate_limiter = PerHostRateLimiter(per_host_rps=per_host_rps, jitter_range=(0.0, 0.0))

    # --- public API -----------------------------------------------------
    def scrape(self, ctx: ScrapeContext) -> Optional[Dict[str, Any]]:
        query = self._build_query(ctx)
        html, final_url = self._retrieve_html(query, ctx.siren)
        if not html:
            return None
        parsed = self._parse_html(html, final_url)
        if not parsed:
            return None
        confidence = self._compute_confidence(ctx, parsed)
        parsed.update(
            {
                "siren": ctx.siren,
                "denomination": ctx.denomination,
                "maps_confidence_score": confidence,
            }
        )
        return parsed

    # --- internals ------------------------------------------------------
    def _build_query(self, ctx: ScrapeContext) -> str:
        parts = [ctx.denomination, ctx.city, ctx.postal_code, "site:maps.google.com"]
        joined = " ".join(part for part in parts if part)
        return joined.strip()

    def _retrieve_html(self, query: str, siren: str) -> Tuple[Optional[str], Optional[str]]:
        if not query:
            return None, None
        encoded = quote_plus(query)
        url = f"https://www.google.com/search?q={encoded}&hl=fr"
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                wait = self._rate_limiter.wait("www.google.com")
                if self._logger and wait > 0:
                    self._logger.debug("rate limiter sleep %.2fs for google.com", wait)
                headers = {
                    "User-Agent": self._ua_pool.get(),
                    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.7,en;q=0.5",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
                with httpx.Client(follow_redirects=True, timeout=self._timeout, proxies=self._proxies) as client:
                    response = client.get(url, headers=headers)
                if response.status_code != httpx.codes.OK:
                    raise RuntimeError(f"unexpected status {response.status_code}")
                html = response.text
                final_url = str(response.url)
                self._persist_html(siren, html)
                if self._delay_range:
                    sleep_with_jitter(self._delay_range)
                return html, final_url
            except Exception as exc:  # pragma: no cover - network errors in prod
                last_exc = exc
                backoff = 1.5 * (attempt + 1)
                if self._logger:
                    self._logger.warning("maps scrape failed (%s), retrying in %.1fs", exc, backoff)
                time.sleep(backoff)
        if self._logger and last_exc:
            self._logger.error("maps scraping failed after retries: %s", last_exc)
        return None, None

    def _persist_html(self, siren: str, html: str) -> None:
        if not self._html_dir:
            return
        ts = int(time.time())
        filename = f"{siren}_{ts}.html"
        io.write_text(self._html_dir / filename, html)

    @staticmethod
    def _parse_html(html: str, final_url: Optional[str]) -> Optional[Dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        business = MapsScraper._extract_jsonld(soup)
        if not business:
            business = MapsScraper._extract_fallback(soup)
        if not business:
            return None
        result = {
            "maps_name": business.get("name") or business.get("title"),
            "address_complete": MapsScraper._normalize_text(MapsScraper._format_address(business.get("address"))),
            "phone": MapsScraper._normalize_text(business.get("telephone")),
            "website": MapsScraper._normalize_text(business.get("url")),
            "reviews_count": MapsScraper._coerce_int(business.get("reviewCount")),
            "rating_avg": MapsScraper._coerce_float(business.get("ratingValue")),
            "google_maps_url": final_url or MapsScraper._normalize_text(business.get("mapsUrl")),
        }
        return result

    @staticmethod
    def _extract_jsonld(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
        for script in scripts:
            raw = script.string or script.get_text()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:  # pragma: no cover - malformed blocks
                continue
            candidates: list[Dict[str, Any]] = []
            if isinstance(data, list):
                candidates.extend(item for item in data if isinstance(item, dict))
            elif isinstance(data, dict):
                if "@graph" in data and isinstance(data["@graph"], list):
                    candidates.extend(item for item in data["@graph"] if isinstance(item, dict))
                candidates.append(data)
            for item in candidates:
                types = item.get("@type")
                if not types:
                    continue
                if isinstance(types, list):
                    type_values = [str(t).lower() for t in types]
                else:
                    type_values = [str(types).lower()]
                if not any("localbusiness" in t or "organization" in t for t in type_values):
                    continue
                agg = item.get("aggregateRating", {}) or {}
                return {
                    "name": item.get("name"),
                    "address": item.get("address"),
                    "telephone": item.get("telephone"),
                    "url": item.get("url") or item.get("sameAs"),
                    "ratingValue": agg.get("ratingValue"),
                    "reviewCount": agg.get("reviewCount"),
                    "mapsUrl": item.get("@id"),
                }
        return None

    @staticmethod
    def _extract_fallback(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        # Minimal fallback using microdata
        container = soup.find(attrs={"itemtype": re.compile("LocalBusiness", re.I)})
        if not container:
            return None
        def _find(prop: str) -> Optional[str]:
            node = container.find(attrs={"itemprop": prop})
            if not node:
                return None
            if node.name == "meta":
                return node.get("content")
            return node.get_text(strip=True)
        address_parts = []
        address_node = container.find(attrs={"itemtype": re.compile("PostalAddress", re.I)})
        if address_node:
            for part in ("streetAddress", "addressLocality", "postalCode", "addressRegion", "addressCountry"):
                value = MapsScraper._normalize_text(MapsScraper._get_itemprop(address_node, part))
                if value:
                    address_parts.append(value)
        return {
            "name": _find("name"),
            "address": ", ".join(address_parts) if address_parts else _find("address"),
            "telephone": _find("telephone"),
            "url": _find("url"),
            "ratingValue": _find("ratingValue"),
            "reviewCount": _find("reviewCount"),
            "mapsUrl": _find("url"),
        }

    @staticmethod
    def _get_itemprop(node: Any, prop: str) -> Optional[str]:
        match = node.find(attrs={"itemprop": prop})
        if not match:
            return None
        if match.name == "meta":
            return match.get("content")
        return match.get_text(strip=True)

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if not value:
            return ""
        text = str(value)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _format_address(address: Any) -> str:
        if not address:
            return ""
        if isinstance(address, str):
            return address
        if isinstance(address, dict):
            parts = [
                address.get("streetAddress"),
                address.get("postalCode"),
                address.get("addressLocality"),
                address.get("addressRegion"),
                address.get("addressCountry"),
            ]
            return ", ".join([MapsScraper._normalize_text(part) for part in parts if part])
        return ""

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(str(value).replace(" ", ""))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _sanitize_for_match(value: str) -> str:
        value = value.lower()
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return value.strip()

    def _compute_confidence(self, ctx: ScrapeContext, parsed: Dict[str, Any]) -> float:
        scores: list[float] = []
        input_name = self._sanitize_for_match(ctx.denomination)
        result_name = self._sanitize_for_match(parsed.get("maps_name", ""))
        if input_name and result_name:
            scores.append(SequenceMatcher(None, input_name, result_name).ratio())

        address_text = parsed.get("address_complete", "") or ""
        address_lower = address_text.lower()
        address_score = 0.0
        if ctx.city and ctx.city.lower() in address_lower:
            address_score += 0.5
        if ctx.postal_code and str(ctx.postal_code) in address_text:
            address_score += 0.5
        if address_score > 0:
            scores.append(min(1.0, address_score))

        if not scores:
            return 0.0
        return round(sum(scores) / len(scores) * 100.0, 2)


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

    if ctx.get("dry_run"):
        empty = pd.DataFrame(columns=[f.name for f in RESULT_SCHEMA])
        empty.to_parquet(parquet_path, index=False)
        io.write_text(jsonl_path, "")
        return {
            "status": "DRY_RUN",
            "file": str(parquet_path),
            "rows": 0,
            "duration_s": round(time.time() - t0, 3),
        }

    input_path = Path(cfg.get("input_path") or outdir / "normalized.parquet")
    if not input_path.exists():
        return {"status": "FAIL", "error": f"missing input parquet: {input_path}"}

    delay_range = tuple(cfg.get("delay_range", DEFAULT_DELAY_RANGE))
    per_host_rps = float(cfg.get("per_host_rps", 1.0))
    timeout = float(cfg.get("timeout", DEFAULT_TIMEOUT))
    batch_size = int(cfg.get("batch_size", DEFAULT_BATCH_SIZE))
    proxies = {
        "http": os.getenv("HTTP_PROXY") or cfg.get("http_proxy"),
        "https": os.getenv("HTTPS_PROXY") or cfg.get("https_proxy"),
    }

    scraper = MapsScraper(
        delay_range=delay_range,
        timeout=timeout,
        proxies=proxies,
        user_agents_path=cfg.get("user_agents_path"),
        max_retries=int(cfg.get("max_retries", 2)),
        logger=logger,
        html_dir=html_dir,
        per_host_rps=per_host_rps,
    )

    columns = ["siren", "denomination", "city", "postal_code"]
    field_order = [field.name for field in RESULT_SCHEMA]
    total_rows = 0
    wrote_rows = False

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
                wrote_rows = True

    if not wrote_rows:
        pd.DataFrame(columns=field_order).to_parquet(parquet_path, index=False)

    return {
        "status": "OK",
        "rows": total_rows,
        "file": str(parquet_path),
        "jsonl": str(jsonl_path),
        "duration_s": round(time.time() - t0, 3),
    }












