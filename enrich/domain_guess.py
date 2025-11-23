#!/usr/bin/env python3
"""
Domain guessing enrichment.

Generate domain variants from company names, validate them with DNS/HTTP,
score them, and emit a joinable parquet with the best guess per company.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import dns.resolver
import httpx
import pandas as pd
import pyarrow as pa

from proxy_manager import ProxyManager
from utils import budget_middleware
from utils.parquet import ParquetBatchWriter, iter_batches

MODULE_LOGGER = logging.getLogger(__name__)
PROXY_MANAGER = ProxyManager()

DOMAIN_GUESS_WORKERS = max(1, int(os.getenv("DOMAIN_GUESS_WORKERS", "200")))
DOMAIN_GUESS_HTTP_TIMEOUT = float(os.getenv("DOMAIN_GUESS_HTTP_TIMEOUT", "6.0"))
DOMAIN_GUESS_DNS_TIMEOUT = float(os.getenv("DOMAIN_GUESS_DNS_TIMEOUT", "3.0"))
DOMAIN_GUESS_MIN_SCORE = float(os.getenv("DOMAIN_GUESS_MIN_SCORE", "0.45"))
DOMAIN_GUESS_FORCE_ALL = os.getenv("DOMAIN_GUESS_FORCE_ALL", "0") == "1"

_tlds_env = os.getenv("DOMAIN_GUESS_TLDS", ".fr,.com,.eu,.net,.org")
DEFAULT_TLDS = [t.strip().lower() for t in _tlds_env.split(",") if t.strip()]
PREFERRED_TLDS_FR = {".fr"}

OUTPUT_FILENAME = "domain_guess.parquet"
USER_AGENT = "Mozilla/5.0 (+domain-guess)"

LEGAL_SUFFIXES = {
    "sarl",
    "sas",
    "sas.",
    "sa",
    "eurl",
    "sasu",
    "sci",
    "sc",
    "holding",
    "gmbh",
    "ag",
    "llc",
    "ltd",
    "inc",
    "sl",
    "srl",
    "plc",
}
GENERIC_PREFIXES = {"cabinet", "ets", "groupe", "societe", "societe.", "ste"}


@dataclass
class DomainCheckResult:
    domain: str
    has_dns: bool = False
    has_mx: bool = False
    status_code: Optional[int] = None
    final_url: str = ""
    snippet: str = ""


def _proxy_settings_for_httpx() -> Optional[Dict[str, str]]:
    if hasattr(PROXY_MANAGER, "as_httpx"):
        return PROXY_MANAGER.as_httpx()  # type: ignore[attr-defined]
    proxies = PROXY_MANAGER.as_requests()
    if not proxies:
        return None
    http_proxy = proxies.get("http")
    https_proxy = proxies.get("https", http_proxy)
    proxy_mapping: Dict[str, str] = {}
    if http_proxy:
        proxy_mapping["http://"] = http_proxy
    if https_proxy:
        proxy_mapping["https://"] = https_proxy
    return proxy_mapping or None


def _strip_accents(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def _tokenize_name(name: str) -> List[str]:
    if not isinstance(name, str):
        return []
    cleaned = _strip_accents(name.lower())
    cleaned = re.sub(r"[^a-z0-9\s\-]", " ", cleaned)
    tokens = [tok for tok in cleaned.replace("-", " ").split() if tok]
    filtered: List[str] = []
    for tok in tokens:
        if tok in LEGAL_SUFFIXES:
            continue
        filtered.append(tok)
    return filtered


def _generate_name_variants(tokens: Sequence[str]) -> List[str]:
    if not tokens:
        return []
    variants: List[str] = []
    joined = "-".join(tokens)
    variants.append(joined)
    variants.append("".join(tokens))
    if len(tokens) > 1:
        variants.append("-".join(tokens[:2]))
        variants.append(tokens[0])
        variants.append("-".join(tokens[-2:]))
    if tokens[0] in GENERIC_PREFIXES and len(tokens) > 1:
        tail = tokens[1:]
        variants.append("-".join(tail))
        variants.append("".join(tail))
    deduped: List[str] = []
    seen = set()
    for v in variants:
        if v and v not in seen:
            deduped.append(v)
            seen.add(v)
    return deduped


def _generate_domains(company_name: str, *, tlds: Sequence[str]) -> List[str]:
    tokens = _tokenize_name(company_name)
    bases = _generate_name_variants(tokens)
    domains: List[str] = []
    for base in bases:
        for tld in tlds:
            domains.append(f"{base}{tld}")
    seen = set()
    unique_domains: List[str] = []
    for d in domains:
        if d not in seen:
            unique_domains.append(d)
            seen.add(d)
    return unique_domains


async def _resolve_domain(domain: str) -> Tuple[bool, bool]:
    resolver = dns.resolver.Resolver()
    resolver.lifetime = DOMAIN_GUESS_DNS_TIMEOUT
    resolver.timeout = DOMAIN_GUESS_DNS_TIMEOUT

    def _resolve(qtype: str) -> bool:
        try:
            resolver.resolve(domain, qtype, lifetime=DOMAIN_GUESS_DNS_TIMEOUT)
            return True
        except dns.resolver.NXDOMAIN:
            return False
        except dns.resolver.NoAnswer:
            return False
        except dns.resolver.NoNameservers:
            return False
        except Exception:
            return False

    has_a = await asyncio.to_thread(_resolve, "A")
    has_aaaa = await asyncio.to_thread(_resolve, "AAAA")
    has_mx = await asyncio.to_thread(_resolve, "MX")
    return (has_a or has_aaaa), has_mx


async def _fetch_domain(
    domain: str,
    client: httpx.AsyncClient,
    *,
    request_tracker=None,
) -> Tuple[Optional[int], str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    for scheme in ("https", "http"):
        url = f"{scheme}://{domain}"
        try:
            response = await client.head(url, follow_redirects=True)
            if request_tracker:
                try:
                    request_tracker(len(response.content or b""))
                except budget_middleware.BudgetExceededError:
                    raise
            status = response.status_code
            if status in (405, 403, 400) or status >= 500:
                response = await client.get(url, headers=headers, follow_redirects=True)
                if request_tracker:
                    try:
                        request_tracker(len(response.content or b""))
                    except budget_middleware.BudgetExceededError:
                        raise
                status = response.status_code
            snippet = ""
            try:
                snippet = (response.text or "")[:1200]
            except Exception:
                snippet = ""
            return status, str(response.url), snippet
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
            continue
        except Exception:
            continue
    return None, "", ""


async def _check_domain(
    domain: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    *,
    request_tracker=None,
) -> DomainCheckResult:
    async with semaphore:
        has_dns, has_mx = await _resolve_domain(domain)
        if not has_dns and not has_mx:
            return DomainCheckResult(domain=domain, has_dns=False, has_mx=False)
        status, final_url, snippet = await _fetch_domain(domain, client, request_tracker=request_tracker)
        return DomainCheckResult(
            domain=domain,
            has_dns=has_dns,
            has_mx=has_mx,
            status_code=status,
            final_url=final_url,
            snippet=(snippet or "")[:1200],
        )


def _select_name_field(row: pd.Series) -> str:
    for key in ("raison_sociale", "denomination", "company_name", "enseigne"):
        if key in row:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _select_country(row: pd.Series) -> str:
    for key in ("pays", "country"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return ""


def _city_tokens(row: pd.Series) -> List[str]:
    for key in ("ville", "commune", "city"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _tokenize_name(value)
    return []


def _status_score(status_code: Optional[int]) -> float:
    if status_code is None:
        return 0.0
    if 200 <= status_code < 300:
        return 0.35
    if 300 <= status_code < 400:
        return 0.3
    if 400 <= status_code < 500:
        return 0.05
    return 0.0


def _score_candidate(
    candidate: str,
    result: DomainCheckResult,
    company_tokens: List[str],
    city_tokens: List[str],
    country: str,
) -> Tuple[float, str]:
    snippet = (result.snippet or "").lower()
    score = 0.0
    parts: List[str] = ["name_variants"]

    if result.has_dns:
        score += 0.25
        parts.append("dns")
    if result.has_mx:
        score += 0.1
    score += _status_score(result.status_code)

    token_hits = [tok for tok in company_tokens if len(tok) >= 4 and (tok in candidate or tok in snippet)]
    if token_hits:
        score += 0.2
        parts.append("name_match")
    if any(tok in snippet for tok in city_tokens if len(tok) >= 4):
        score += 0.05
        parts.append("city_match")

    for preferred in PREFERRED_TLDS_FR:
        if country in ("FR", "FRA", "FRANCE") and candidate.endswith(preferred):
            score += 0.05
            parts.append("tld_pref")
            break

    return min(score, 1.0), "+".join(parts)


async def _validate_domains(
    domains: Iterable[str],
    *,
    request_tracker=None,
) -> Dict[str, DomainCheckResult]:
    proxies = _proxy_settings_for_httpx()
    limits = httpx.Limits(max_connections=DOMAIN_GUESS_WORKERS)
    timeout = httpx.Timeout(DOMAIN_GUESS_HTTP_TIMEOUT)
    semaphore = asyncio.Semaphore(DOMAIN_GUESS_WORKERS)
    results: Dict[str, DomainCheckResult] = {}
    unique_domains = list(set(domains))

    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=timeout, limits=limits, proxies=proxies) as client:
        tasks: Dict[str, asyncio.Task] = {
            domain: asyncio.create_task(
                _check_domain(domain, client, semaphore, request_tracker=request_tracker)
            )
            for domain in unique_domains
        }

        for domain, task in tasks.items():
            try:
                result = await task
                results[domain] = result
            except budget_middleware.BudgetExceededError:
                raise
            except Exception as exc:  # pragma: no cover - resilience
                MODULE_LOGGER.warning("Domain guess: failed to validate %s: %s", domain, exc)
                results[domain] = DomainCheckResult(domain=domain)

    return results


def _to_json(value: object) -> Optional[str]:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return None


def _iter_input_batches(path: Path) -> Iterable[pd.DataFrame]:
    if path.suffix.lower() == ".csv":
        for chunk in pd.read_csv(path, chunksize=5000):
            yield chunk
    else:
        yield from iter_batches(path)


def run(cfg: dict, ctx: dict) -> dict:
    t0 = pd.Timestamp.utcnow()
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    input_path = outdir / "normalized.parquet"
    if not input_path.exists():
        input_path = outdir / "normalized.csv"
        if not input_path.exists():
            return {"status": "FAIL", "error": f"missing normalized input in {outdir}"}

    output_path = outdir / OUTPUT_FILENAME
    logger = ctx.get("logger") or MODULE_LOGGER
    request_tracker = ctx.get("request_tracker")

    logger.info(
        "Domain guess config | workers=%d http_timeout=%.1fs dns_timeout=%.1fs min_score=%.2f force_all=%s tlds=%s",
        DOMAIN_GUESS_WORKERS,
        DOMAIN_GUESS_HTTP_TIMEOUT,
        DOMAIN_GUESS_DNS_TIMEOUT,
        DOMAIN_GUESS_MIN_SCORE,
        DOMAIN_GUESS_FORCE_ALL,
        ",".join(DEFAULT_TLDS),
    )
    logger.info("Domain guess proxy enabled: %s", PROXY_MANAGER.enabled)

    total_rows = 0
    needs_guess = 0
    filled = 0
    scores: List[float] = []
    domain_cache: Dict[str, DomainCheckResult] = {}

    key_columns = ("siren", "siret", "id", "company_id")

    with ParquetBatchWriter(output_path) as writer:
        for pdf in _iter_input_batches(input_path):
            if pdf.empty:
                continue
            total_rows += len(pdf)

            pdf = pdf.copy()
            for key in key_columns:
                if key not in pdf.columns:
                    pdf[key] = pd.NA
                pdf[key] = pdf[key].astype("string")

            # Prepare target columns
            pdf["domain_candidate_best"] = pd.NA
            pdf["domain_candidates"] = pd.NA
            pdf["domain_guess_score"] = pd.NA
            pdf["domain_guess_source"] = pd.NA

            existing_domain = pd.Series("", index=pdf.index, dtype="string")
            for column in ("domain", "domain_root", "site_web", "siteweb", "website"):
                if column in pdf.columns:
                    col = pdf[column].fillna("").astype("string")
                    existing_domain = existing_domain.where(existing_domain != "", col)

            if DOMAIN_GUESS_FORCE_ALL:
                mask = pd.Series(True, index=pdf.index)
            else:
                mask = existing_domain.str.strip().eq("")

            batch_candidates: Dict[int, List[str]] = {}
            batch_context: Dict[int, Tuple[List[str], List[str], str]] = {}
            unique_domains: List[str] = []

            for idx, row in pdf[mask].iterrows():
                name = _select_name_field(row)
                if not name:
                    continue
                company_tokens = _tokenize_name(name)
                if not company_tokens:
                    continue
                country = _select_country(row)
                city_tokens = _city_tokens(row)
                candidates = _generate_domains(name, tlds=DEFAULT_TLDS)
                if not candidates:
                    continue
                needs_guess += 1
                batch_candidates[idx] = candidates
                batch_context[idx] = (company_tokens, city_tokens, country)
                for cand in candidates:
                    if cand not in domain_cache:
                        unique_domains.append(cand)

            if unique_domains:
                try:
                    results = asyncio.run(_validate_domains(unique_domains, request_tracker=request_tracker))
                    domain_cache.update(results)
                except budget_middleware.BudgetExceededError:
                    raise
                except Exception as exc:
                    logger.error("Domain guess validation failed: %s", exc, exc_info=True)

            for idx, candidates in batch_candidates.items():
                company_tokens, city_tokens, country = batch_context[idx]
                scored: List[Tuple[str, float, str, DomainCheckResult]] = []
                for cand in candidates:
                    result = domain_cache.get(cand) or DomainCheckResult(domain=cand)
                    score, source = _score_candidate(cand, result, company_tokens, city_tokens, country)
                    scored.append((cand, score, source, result))
                scored.sort(key=lambda x: x[1], reverse=True)
                if scored:
                    pdf.at[idx, "domain_candidates"] = _to_json(
                        [
                            {
                                "domain": cand,
                                "score": round(score, 4),
                                "status": res.status_code,
                                "dns": res.has_dns or res.has_mx,
                            }
                            for cand, score, _, res in scored[:10]
                        ]
                    )
                    best_cand, best_score, best_source, _ = scored[0]
                    if best_score >= DOMAIN_GUESS_MIN_SCORE:
                        pdf.at[idx, "domain_candidate_best"] = best_cand
                        pdf.at[idx, "domain_guess_score"] = float(best_score)
                        pdf.at[idx, "domain_guess_source"] = best_source
                        filled += 1
                        scores.append(best_score)

            table = pa.Table.from_pandas(pdf, preserve_index=False)
            writer.write_table(table)

    elapsed = (pd.Timestamp.utcnow() - t0).total_seconds()
    avg_score = float(sum(scores) / len(scores)) if scores else 0.0
    fill_rate = (filled / needs_guess * 100) if needs_guess else 0.0

    logger.info(
        "domain_guess summary | rows=%d | missing_input=%d | filled=%d (%.1f%%) | avg_score=%.3f | output=%s | elapsed=%.1fs",
        total_rows,
        needs_guess,
        filled,
        fill_rate,
        avg_score,
        output_path,
        elapsed,
    )

    return {
        "status": "OK",
        "rows": total_rows,
        "missing_input": needs_guess,
        "filled": filled,
        "avg_score": avg_score,
        "output": str(output_path),
        "elapsed": elapsed,
    }
