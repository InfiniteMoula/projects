"""Headless fallback collection for domains without detected contacts."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
from readability import Document

from utils import io
from utils.ua import load_user_agent_pool
from utils.url import canonicalize, looks_like_home
from bs4 import BeautifulSoup

try:
    from playwright.async_api import async_playwright, Error as PlaywrightError
except ImportError:  # pragma: no cover - handled in run()
    async_playwright = None  # type: ignore
    PlaywrightError = Exception  # type: ignore

MAX_PAGE_BYTES = 1_000_000
HTML_TRUNCATE_CHARS = 100_000
TARGET_PATHS = (
    "/",
    "/contact",
    "/contacts",
    "/nous-contacter",
    "/mentions-legales",
    "/mentions_l\u00e9gales",
    "/mentions-legales/",
    "/a-propos",
    "/about",
    "/legal",
    "/impressum",
)
DEFAULT_TIMEOUT = 10.0


@dataclass
class TargetRow:
    domain: str
    siren: Optional[str]
    denomination: Optional[str]
    top_url: Optional[str]


def _extract_text(html: str) -> str:
    try:
        summary_html = Document(html).summary()
    except Exception:
        summary_html = html
    parsed = BeautifulSoup(summary_html, "lxml")
    text = parsed.get_text(" ", strip=True)
    return text[:200000]


def _load_targets(outdir: Path, logger) -> List[TargetRow]:
    fallback_path = outdir / "contacts" / "no_contact.csv"
    if not fallback_path.exists():
        if logger:
            logger.info("collect_headless_fallback: no no_contact.csv found, nothing to do")
        return []
    try:
        df = pd.read_csv(fallback_path)
    except Exception as exc:
        if logger:
            logger.warning("collect_headless_fallback: unable to read %s: %s", fallback_path, exc)
        return []
    required = {"domain"}
    if not required.issubset(df.columns):
        if logger:
            logger.warning("collect_headless_fallback: missing domain column in %s", fallback_path)
        return []
    targets: List[TargetRow] = []
    for _, row in df.iterrows():
        domain = str(row.get("domain") or "").strip().lower()
        if not domain:
            continue
        siren = str(row.get("siren") or "").strip() or None
        denomination = str(row.get("denomination") or "").strip() or None
        top_url = str(row.get("top_url") or "").strip() or None
        targets.append(TargetRow(domain=domain, siren=siren, denomination=denomination, top_url=top_url))
    return targets


def _candidate_urls(target: TargetRow) -> List[str]:
    bases: List[str] = []
    if target.top_url:
        bases.append(target.top_url)
    bases.append(f"https://{target.domain}")
    bases.append(f"http://{target.domain}")
    urls: List[str] = []
    for base in dict.fromkeys(bases):
        if not base:
            continue
        base_norm = canonicalize(base)
        if not base_norm:
            continue
        for path in TARGET_PATHS:
            if path == "/" and not looks_like_home(base_norm):
                urls.append(base_norm)
            elif path == "/":
                urls.append(base_norm)
            else:
                candidate = canonicalize(base_norm.rstrip("/") + path)
                urls.append(candidate)
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


async def _render_page(context, url: str, timeout: float, siren_list: Sequence[str], denom_list: Sequence[str], domain: str):
    page = await context.new_page()
    page.set_default_timeout(timeout * 1000)
    try:
        response = await page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
    except Exception:
        await page.close()
        return None
    if response is None:
        await page.close()
        return None
    try:
        html = await page.content()
    except Exception:
        await page.close()
        return None
    text_content = _extract_text(html)
    final_url = str(page.url)
    await page.close()
    content_bytes = html.encode("utf-8", errors="replace")
    if len(content_bytes) > MAX_PAGE_BYTES:
        content_bytes = content_bytes[:MAX_PAGE_BYTES]
    entry = {
        "domain": domain,
        "requested_url": url,
        "url": final_url,
        "status": response.status,
        "content_type": "text/html",
        "bytes": len(content_bytes),
        "discovered_at": io.now_iso(),
        "siren_list": list(siren_list),
        "denominations": list(denom_list),
        "https": final_url.startswith("https"),
        "content_text": text_content,
        "content_html_trunc": html[:HTML_TRUNCATE_CHARS],
        "rendered": True,
    }
    return entry


async def _collect_async(targets: Iterable[TargetRow], timeout: float, max_pages: int, user_agent: str, logger):
    if async_playwright is None:
        return []
    collected: List[dict] = []
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True, args=["--disable-gpu"])
        except PlaywrightError as exc:  # pragma: no cover - depends on environment
            if logger:
                logger.warning("collect_headless_fallback: unable to launch Chromium: %s", exc)
            return []
        context = await browser.new_context(user_agent=user_agent, java_script_enabled=True)
        context.set_default_navigation_timeout(timeout * 1000)

        async def _route_handler(route):
            if route.request.resource_type in {"image", "stylesheet", "font", "media"}:
                await route.abort()
            else:
                await route.continue_()

        await context.route("**/*", _route_handler)
        try:
            for target in targets:
                siren_list = [target.siren] if target.siren else []
                denom_list = [target.denomination] if target.denomination else []
                pages_for_target = 0
                for url in _candidate_urls(target):
                    if pages_for_target >= max_pages:
                        break
                    entry = await _render_page(
                        context,
                        url,
                        timeout=timeout,
                        siren_list=siren_list,
                        denom_list=denom_list,
                        domain=target.domain,
                    )
                    if entry:
                        collected.append(entry)
                        pages_for_target += 1
        finally:
            await context.close()
            await browser.close()
    return collected


def run(cfg: dict, ctx: dict) -> dict:
    logger = ctx.get("logger")
    if async_playwright is None:
        message = "collect_headless_fallback: Playwright is not installed; skipping headless fallback."
        if logger:
            logger.info(message)
        return {"status": "SKIPPED", "reason": "PLAYWRIGHT_MISSING"}

    outdir = Path(ctx["outdir"])
    targets = _load_targets(outdir, logger)

    if not targets:
        return {"status": "SKIPPED", "reason": "NO_TARGETS"}

    sample = int(ctx.get("sample") or 0)
    if sample > 0:
        targets = targets[:sample]

    headless_cfg = cfg.get("headless", {})
    timeout = float(headless_cfg.get("timeout_s") or DEFAULT_TIMEOUT)
    max_pages = int(headless_cfg.get("max_pages_per_domain") or 3)

    ua_pool = ctx.get("user_agent_pool") or load_user_agent_pool(None)
    user_agent = ctx.get("crawler_user_agent") or ua_pool.get()

    pages = asyncio.run(_collect_async(targets, timeout=timeout, max_pages=max_pages, user_agent=user_agent, logger=logger))

    if not pages:
        return {"status": "SKIPPED", "reason": "NO_PAGES"}

    headless_dir = io.ensure_dir(outdir / "headless")
    output_path = headless_dir / "pages_dynamic.parquet"
    pd.DataFrame(pages).to_parquet(output_path, index=False)

    if logger:
        logger.info("collect_headless_fallback: collected %d rendered pages", len(pages))

    return {
        "status": "OK",
        "output": str(output_path),
        "pages": len(pages),
    }
