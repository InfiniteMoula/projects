#!/usr/bin/env python3
"""
Enrichment step: scrape phone numbers from company websites.

Uses existing/guessed domains, hits a handful of high-value pages,
extracts phone numbers, scores them, and writes a joinable parquet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
import pandas as pd
import pyarrow as pa

from proxy_manager import ProxyManager
from utils import budget_middleware
from utils.parquet import ParquetBatchWriter, iter_batches

MODULE_LOGGER = logging.getLogger(__name__)
PROXY_MANAGER = ProxyManager()

PHONE_SITE_WORKERS = max(1, int(os.getenv("PHONE_SITE_WORKERS", "50")))
PHONE_SITE_TIMEOUT = float(os.getenv("PHONE_SITE_TIMEOUT", "6.0"))
PHONE_SITE_MAX_PAGES_PER_DOMAIN = max(1, int(os.getenv("PHONE_SITE_MAX_PAGES_PER_DOMAIN", "4")))
PHONE_SITE_MIN_SCORE = float(os.getenv("PHONE_SITE_MIN_SCORE", "0.4"))
PHONE_SITE_FORCE_ALL = os.getenv("PHONE_SITE_FORCE_ALL", "0") == "1"
PHONE_SITE_USER_AGENT = os.getenv("PHONE_SITE_USER_AGENT", "Mozilla/5.0 (+phone-site)")
PHONE_SITE_URL_PATHS = os.getenv(
    "PHONE_SITE_URL_PATHS",
    "/,/contact,/contact/,/contactez-nous,/contactez-nous/,/mentions-legales,/mentions-legales/,/mentions,/a-propos,/nous-contacter",
)
URL_SUFFIXES: List[str] = [p.strip() for p in PHONE_SITE_URL_PATHS.split(",") if p.strip()]

KEY_COLUMNS = ("siren", "siret", "id", "company_id")
DOMAIN_COLUMNS = ("domain", "domain_root", "site_web", "siteweb", "website")
PHONE_COLUMNS = ("telephone_norm", "phone", "telephone", "tel")

PHONE_REGEXES = [
    re.compile(r"\+33\s*(\(0\))?\s*[1-9](?:[\s\.-]?\d{2}){4}"),
    re.compile(r"0[1-9](?:[\s\.-]?\d{2}){4}"),
    re.compile(r"\+?\d[\d\s\.-]{8,16}\d"),
]

CONTEXT_HINTS = ("tel", "tél", "téléphone", "contact", "call", "appel")
FAX_HINTS = ("fax",)


def _proxy_settings_for_httpx() -> Optional[Dict[str, str]]:
    if hasattr(PROXY_MANAGER, "as_httpx"):
        return PROXY_MANAGER.as_httpx()  # type: ignore[attr-defined]
    proxies = PROXY_MANAGER.as_requests()
    if not proxies:
        return None
    http_proxy = proxies.get("http")
    https_proxy = proxies.get("https", http_proxy)
    mapping: Dict[str, str] = {}
    if http_proxy:
        mapping["http://"] = http_proxy
    if https_proxy:
        mapping["https://"] = https_proxy
    return mapping or None


def _normalize_domain(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    text = raw.strip().lower()
    if not text:
        return ""
    if text.startswith("http://") or text.startswith("https://"):
        text = text.split("://", 1)[1]
    if "/" in text:
        text = text.split("/", 1)[0]
    if text.startswith("www."):
        text = text[4:]
    return text.strip()


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _normalize_phone(raw: str) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    txt = raw.strip()
    if not txt:
        return None
    txt = txt.replace("\xa0", " ")
    digits_only = re.sub(r"[^\d+]", "", txt)
    if digits_only.startswith("+33"):
        digits_only = "0" + digits_only[3:]
    digits = re.sub(r"[^\d]", "", digits_only)
    if digits.startswith("0033"):
        digits = "0" + digits[4:]
    if digits.startswith("33") and len(digits) == 11:
        digits = "0" + digits[2:]
    if len(digits) == 10 and digits.startswith("0"):
        return digits
    if len(digits) >= 9 and len(digits) <= 12:
        return digits
    return None


def _score_candidate(phone_norm: str, url: str, context: str) -> float:
    score = 0.35
    path = url.lower()
    if any(seg in path for seg in ("contact", "contacter", "nous-contacter")):
        score += 0.2
    elif any(seg in path for seg in ("mentions", "legal", "legales")):
        score += 0.1
    else:
        score += 0.05  # home/default page
    ctx_low = context.lower()
    if any(hint in ctx_low for hint in CONTEXT_HINTS):
        score += 0.1
    if any(hint in ctx_low for hint in FAX_HINTS):
        score -= 0.1
    if phone_norm.startswith("06") or phone_norm.startswith("07"):
        score += 0.05  # FR mobile
    return max(0.0, min(score, 1.0))


def _extract_phones(html: str, url: str) -> List[Dict]:
    results: List[Dict] = []
    if not html:
        return results
    for regex in PHONE_REGEXES:
        for match in regex.finditer(html):
            raw = match.group(0)
            norm = _normalize_phone(raw)
            if not norm:
                continue
            start, end = match.start(), match.end()
            context_window = html[max(0, start - 80) : min(len(html), end + 80)]
            context_clean = _strip_accents(context_window)
            score = _score_candidate(norm, url, context_clean)
            results.append(
                {
                    "phone_raw": raw.strip(),
                    "phone_normalized": norm,
                    "score": round(score, 4),
                    "url": url,
                    "context": context_clean[:160],
                }
            )
    # Deduplicate by normalized phone, keep best score
    dedup: Dict[str, Dict] = {}
    for item in results:
        key = item["phone_normalized"]
        existing = dedup.get(key)
        if not existing or item["score"] > existing["score"]:
            dedup[key] = item
    return sorted(dedup.values(), key=lambda x: x["score"], reverse=True)


async def _fetch_url(
    url: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    request_tracker=None,
) -> str:
    async with semaphore:
        try:
            resp = await client.get(url, follow_redirects=True)
            if request_tracker:
                try:
                    request_tracker(len(resp.content or b""))
                except budget_middleware.BudgetExceededError:
                    raise
            if resp.status_code >= 400:
                return ""
            return resp.text or ""
        except budget_middleware.BudgetExceededError:
            raise
        except httpx.HTTPError as exc:
            if request_tracker:
                resp_obj = getattr(exc, "response", None)
                size = len(resp_obj.content or b"") if resp_obj and resp_obj.content else 0
                try:
                    request_tracker(size)
                except budget_middleware.BudgetExceededError:
                    raise
            return ""
        except Exception:
            return ""


async def _fetch_domain_pages(
    domain: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    request_tracker=None,
) -> List[Dict]:
    urls: List[str] = []
    for suffix in URL_SUFFIXES[:PHONE_SITE_MAX_PAGES_PER_DOMAIN]:
        if not suffix.startswith("/"):
            suffix = "/" + suffix
        urls.append(f"https://{domain}{suffix}")
    results: List[Dict] = []
    for url in urls:
        html = await _fetch_url(url, client, semaphore, request_tracker=request_tracker)
        if not html and url.startswith("https://"):
            fallback = url.replace("https://", "http://", 1)
            html = await _fetch_url(fallback, client, semaphore, request_tracker=request_tracker)
            url = fallback
        if not html:
            continue
        phones = _extract_phones(html, url)
        results.extend(phones)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


async def _gather_domain_candidates(
    domains: Iterable[str],
    *,
    request_tracker=None,
) -> Dict[str, List[Dict]]:
    proxies = _proxy_settings_for_httpx()
    limits = httpx.Limits(max_connections=PHONE_SITE_WORKERS)
    timeout = httpx.Timeout(PHONE_SITE_TIMEOUT)
    semaphore = asyncio.Semaphore(PHONE_SITE_WORKERS)
    domain_set = list(set(domains))
    out: Dict[str, List[Dict]] = {}
    async with httpx.AsyncClient(
        headers={"User-Agent": PHONE_SITE_USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
        timeout=timeout,
        limits=limits,
        proxies=proxies,
    ) as client:
        tasks = {
            asyncio.create_task(_fetch_domain_pages(domain, client, semaphore, request_tracker=request_tracker)): domain
            for domain in domain_set
        }
        for task in asyncio.as_completed(tasks):
            domain = tasks[task]
            try:
                out[domain] = await task
            except budget_middleware.BudgetExceededError:
                raise
            except Exception as exc:
                MODULE_LOGGER.warning("phone_site: failed to fetch %s: %s", domain, exc)
                out[domain] = []
    return out


def _iter_input_batches(path: Path) -> Iterable[pd.DataFrame]:
    if path.suffix.lower() == ".csv":
        for chunk in pd.read_csv(path, chunksize=5000):
            yield chunk
    else:
        yield from iter_batches(path)


def _load_domain_guess(outdir: Path) -> Dict[str, str]:
    guess_path = outdir / "domain_guess.parquet"
    if not guess_path.exists():
        return {}
    mapping: Dict[str, str] = {}
    for pdf in iter_batches(guess_path, columns=list(KEY_COLUMNS) + ["domain_candidate_best"]):
        for _, row in pdf.iterrows():
            domain = str(row.get("domain_candidate_best") or "").strip()
            if not domain:
                continue
            norm = _normalize_domain(domain)
            if not norm:
                continue
            key = None
            for col in KEY_COLUMNS:
                val = str(row.get(col) or "").strip()
                if val:
                    key = f"{col}:{val}"
                    break
            if key:
                mapping[key] = norm
    return mapping


def _resolve_key(row: pd.Series) -> Optional[str]:
    for col in KEY_COLUMNS:
        val = row.get(col)
        if isinstance(val, str):
            if val.strip():
                return f"{col}:{val.strip()}"
        elif pd.notna(val):
            text = str(val).strip()
            if text:
                return f"{col}:{text}"
    return None


def run(cfg: dict, ctx: dict) -> dict:
    t_start = time.time()
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    input_path = outdir / "normalized.parquet"
    if not input_path.exists():
        input_path = outdir / "normalized.csv"
        if not input_path.exists():
            return {"status": "FAIL", "error": f"missing normalized input in {outdir}"}

    output_path = outdir / "phone_site.parquet"
    logger = ctx.get("logger") or MODULE_LOGGER
    request_tracker = ctx.get("request_tracker")

    logger.info(
        "Phone site config | workers=%d timeout=%.1fs max_pages=%d min_score=%.2f force_all=%s paths=%s",
        PHONE_SITE_WORKERS,
        PHONE_SITE_TIMEOUT,
        PHONE_SITE_MAX_PAGES_PER_DOMAIN,
        PHONE_SITE_MIN_SCORE,
        PHONE_SITE_FORCE_ALL,
        ",".join(URL_SUFFIXES),
    )
    logger.info("Phone site proxy enabled: %s", PROXY_MANAGER.enabled)

    domain_guess_map = _load_domain_guess(outdir)

    total_rows = 0
    eligible_rows = 0
    filled_rows = 0
    scores: List[float] = []

    with ParquetBatchWriter(output_path) as writer:
        for pdf in _iter_input_batches(input_path):
            if pdf.empty:
                continue
            total_rows += len(pdf)
            pdf = pdf.copy()

            for key in KEY_COLUMNS:
                if key not in pdf.columns:
                    pdf[key] = pd.NA
                pdf[key] = pdf[key].astype("string")

            # Ensure output columns exist (preserve if already present)
            if "phone_site_best" not in pdf.columns:
                pdf["phone_site_best"] = pd.NA
            if "phone_site_candidates" not in pdf.columns:
                pdf["phone_site_candidates"] = pd.NA
            if "phone_site_score" not in pdf.columns:
                pdf["phone_site_score"] = pd.NA
            if "phone_site_source" not in pdf.columns:
                pdf["phone_site_source"] = pd.NA

            existing_phone = pd.Series("", index=pdf.index, dtype="string")
            for col in PHONE_COLUMNS:
                if col in pdf.columns:
                    series = pdf[col].fillna("").astype("string")
                    existing_phone = existing_phone.where(existing_phone != "", series)
            if "phone_site_best" in pdf.columns:
                existing_phone = existing_phone.where(existing_phone != "", pdf["phone_site_best"].fillna("").astype("string"))

            domains_for_row: Dict[int, str] = {}
            for idx, row in pdf.iterrows():
                if not PHONE_SITE_FORCE_ALL and existing_phone.loc[idx].strip():
                    continue
                domain_val = ""
                for col in DOMAIN_COLUMNS:
                    if col in row and isinstance(row[col], str) and row[col].strip():
                        domain_val = row[col].strip()
                        break
                    if col in row and pd.notna(row[col]) and str(row[col]).strip():
                        domain_val = str(row[col]).strip()
                        break
                if not domain_val:
                    key = _resolve_key(row)
                    if key and key in domain_guess_map:
                        domain_val = domain_guess_map[key]
                norm_domain = _normalize_domain(domain_val)
                if not norm_domain:
                    continue
                domains_for_row[idx] = norm_domain

            if not domains_for_row:
                table = pa.Table.from_pandas(pdf, preserve_index=False)
                writer.write_table(table)
                continue

            eligible_rows += len(domains_for_row)
            unique_domains = set(domains_for_row.values())

            try:
                domain_candidates = asyncio.run(_gather_domain_candidates(unique_domains, request_tracker=request_tracker))
            except budget_middleware.BudgetExceededError:
                raise
            except Exception as exc:
                logger.error("phone_site: domain fetch failed: %s", exc, exc_info=True)
                domain_candidates = {d: [] for d in unique_domains}

            for idx, domain in domains_for_row.items():
                candidates = domain_candidates.get(domain, [])
                if candidates:
                    best = candidates[0]
                    if best["score"] >= PHONE_SITE_MIN_SCORE:
                        pdf.at[idx, "phone_site_best"] = best["phone_normalized"]
                        pdf.at[idx, "phone_site_score"] = float(best["score"])
                        pdf.at[idx, "phone_site_source"] = best.get("url")
                        filled_rows += 1
                        scores.append(float(best["score"]))
                if candidates:
                    pdf.at[idx, "phone_site_candidates"] = json.dumps(candidates[:10], ensure_ascii=False)

            table = pa.Table.from_pandas(pdf, preserve_index=False)
            writer.write_table(table)

    elapsed = time.time() - t_start
    fill_rate = (filled_rows / eligible_rows * 100) if eligible_rows else 0.0
    avg_score = float(sum(scores) / len(scores)) if scores else 0.0

    logger.info(
        "phone_site summary | rows=%d | eligible=%d | filled=%d (%.1f%%) | avg_score=%.3f | output=%s | elapsed=%.1fs",
        total_rows,
        eligible_rows,
        filled_rows,
        fill_rate,
        avg_score,
        output_path,
        elapsed,
    )

    return {
        "status": "OK",
        "rows": total_rows,
        "eligible": eligible_rows,
        "filled": filled_rows,
        "fill_rate": fill_rate,
        "avg_score": avg_score,
        "output": str(output_path),
        "elapsed": elapsed,
    }
