"""Optional Scrapy benchmark spider to compare contact extraction performance."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd

from utils import io


def _load_targets(outdir: Path, sample: int = 0) -> List[str]:
    serp_path = outdir / "serp" / "serp_results.parquet"
    if not serp_path.exists():
        raise FileNotFoundError(f"SERP results not found at {serp_path}")
    df = pd.read_parquet(serp_path)
    domains = df.get("domain") or df.get("top_domain")
    if domains is None:
        return []
    urls = [f"https://{str(domain).strip()}" for domain in domains.dropna()]
    seen: Set[str] = set()
    ordered = []
    for url in urls:
        if url and url not in seen:
            seen.add(url)
            ordered.append(url)
            if sample and len(ordered) >= sample:
                break
    return ordered


async def _run_scrapy(urls: Iterable[str], output_path: Path) -> dict:
    try:
        import scrapy  # type: ignore
        from scrapy.crawler import CrawlerRunner  # type: ignore
        from scrapy.signalmanager import dispatcher  # type: ignore
        from twisted.internet import reactor  # type: ignore
        from twisted.internet.defer import Deferred  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Scrapy is not installed; install `scrapy` to use this benchmark") from exc

    collected = []
    start_time = time.time()

    class ContactSpider(scrapy.Spider):  # type: ignore
        name = "contact_benchmark"

        custom_settings = {
            "LOG_ENABLED": False,
            "DOWNLOAD_TIMEOUT": 10,
        }

        def __init__(self, start_urls: List[str]):
            super().__init__()
            self.start_urls = start_urls
            self.pages = 0

        def parse(self, response):
            self.pages += 1
            emails = set(response.css("a[href^='mailto:']::attr(href)").getall())
            phones = set(response.css("a[href^='tel:']::attr(href)").getall())
            for mail in response.css("::text").re(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"):
                emails.add(mail)
            for tel in response.css("::text").re(r"\+?\d[\d\s().-]{7,}"):
                phones.add(tel)
            collected.append(
                {
                    "url": response.url,
                    "status": response.status,
                    "emails": sorted(set(e.replace("mailto:", "").lower() for e in emails)),
                    "phones": sorted(set(p.replace("tel:", "") for p in phones)),
                }
            )
            contact_links = []
            for href in response.css("a::attr(href)").getall():
                href_lower = href.lower()
                if any(token in href_lower for token in ["contact", "mention", "legal", "about", "propos"]):
                    contact_links.append(response.urljoin(href))
            for link in contact_links[:5]:
                yield scrapy.Request(link, callback=self.parse_contact)

        def parse_contact(self, response):
            yield from self.parse(response)

    runner = CrawlerRunner()
    crawler = runner.create_crawler(ContactSpider(start_urls=list(urls)))

    items = {"pages": 0}

    def _item_scraped(item):
        items["pages"] += 1

    dispatcher.connect(_item_scraped, signal=scrapy.signals.item_scraped)  # type: ignore

    deferred: Deferred = runner.crawl(crawler)
    deferred.addBoth(lambda _: reactor.stop())  # type: ignore
    reactor.run()  # type: ignore
    elapsed = time.time() - start_time

    io.ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(collected, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "pages": crawler.spider.pages,
        "elapsed": elapsed,
        "pages_per_sec": crawler.spider.pages / elapsed if elapsed > 0 else 0.0,
        "item_count": len(collected),
    }


def run(cfg: dict, ctx: dict) -> dict:
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    sample = int(ctx.get("sample") or cfg.get("benchmark", {}).get("sample") or 0)
    urls = _load_targets(outdir, sample=sample)
    if not urls:
        return {"status": "SKIPPED", "reason": "NO_URLS"}

    output_path = outdir / "experimental" / "scrapy_output.json"
    try:
        stats = asyncio.run(_run_scrapy(urls, output_path))
    except RuntimeError as exc:
        return {"status": "SKIPPED", "reason": str(exc)}

    return {
        "status": "OK",
        "urls": len(urls),
        "output": str(output_path),
        **stats,
    }


__all__ = ["run"]

