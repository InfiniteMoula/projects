
# FILE: enrich/site_probe.py
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import pyarrow as pa
import requests

from utils import budget_middleware
from utils.state import SequentialRunState
from utils.parquet import ParquetBatchWriter, iter_batches
from proxy_manager import ProxyManager

REQ_TIMEOUT = 4.0
UA = "Mozilla/5.0 (+contact-probe)"
DEFAULT_CACHE_TTL = 900
DEFAULT_RETRIES = 2
MODULE_LOGGER = logging.getLogger(__name__)
PROXY_MANAGER = ProxyManager()


def _probe(
    url: str,
    *,
    timeout: float,
    request_tracker: Optional[Callable[[int], None]] = None,
) -> tuple[bool, str]:
    if not isinstance(url, str) or not url.strip():
        return False, ""
    u = url.strip()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    try:
        kwargs = {
            "allow_redirects": True,
            "timeout": timeout,
            "headers": {"User-Agent": UA},
        }
        proxies = PROXY_MANAGER.as_requests()
        if proxies:
            kwargs["proxies"] = proxies
        response = requests.head(u, **kwargs)
        if request_tracker:
            try:
                content_length = int(response.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                content_length = 0
            request_tracker(content_length)
        ok = 200 <= response.status_code < 400
        return ok, response.url if ok else ""
    except budget_middleware.BudgetExceededError:
        raise
    except Exception as exc:
        if PROXY_MANAGER.enabled:
            MODULE_LOGGER.warning("Proxy HEAD request failed for %s: %s", u, exc, exc_info=True)
        if request_tracker:
            try:
                request_tracker(0)
            except budget_middleware.BudgetExceededError:
                raise
        return False, ""


def _probe_with_retry(
    url: str,
    *,
    timeout: float,
    retries: int,
    request_tracker: Optional[Callable[[int], None]] = None,
) -> tuple[bool, str]:
    last: tuple[bool, str] = (False, "")
    for _ in range(max(retries, 1)):
        last = _probe(url, timeout=timeout, request_tracker=request_tracker)
        if last[0]:
            break
    return last


def run(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    
    # Primary input: Google Maps enriched data
    maps_inp = outdir / "google_maps_enriched.parquet"
    # Fallback input: domain enriched data
    domain_inp = outdir / "enriched_domain.parquet"
    
    inp = maps_inp if maps_inp.exists() else domain_inp
    outp = outdir / "enriched_site.parquet"
    if not inp.exists():
        return {"status": "WARN", "error": f"missing input: {inp}"}

    timeout = float(ctx.get("site_probe_timeout", REQ_TIMEOUT))
    ttl = int(ctx.get("site_probe_ttl", DEFAULT_CACHE_TTL))
    retries = int(ctx.get("site_probe_retries", DEFAULT_RETRIES))
    workers = max(1, int(ctx.get("workers", 4)))
    dry_run = bool(ctx.get("dry_run", False))
    logger = ctx.get("logger")
    using_maps_data = inp == maps_inp
    request_tracker = ctx.get("request_tracker")

    if logger:
        logger.info(f"Site probe using {'Google Maps' if using_maps_data else 'domain'} data")

    cache: Dict[str, Tuple[float, tuple[bool, str]]] = {}

    def probe_cached(raw_url: str) -> tuple[bool, str]:
        if dry_run:
            return False, ""
        key = raw_url.strip().lower()
        now = time.time()
        result = cache.get(key)
        if result and result[0] > now:
            return result[1]
        value = _probe_with_retry(raw_url, timeout=timeout, retries=retries, request_tracker=request_tracker)
        cache[key] = (now + ttl, value)
        return value

    state = SequentialRunState(outdir / "site_probe_state.json")
    seen_urls = set(state.completed)
    seen_urls.update(state.failed.keys())
    total_known = int(state.metadata.get("total") or len(seen_urls))
    if not state.metadata.get("total"):
        state.set_metadata(total=total_known)

    total = 0

    try:
        with ParquetBatchWriter(outp) as writer:
            for pdf in iter_batches(inp):
                if pdf.empty:
                    continue

                pdf["siteweb"] = pdf.get("siteweb", pd.Series(pd.NA, index=pdf.index)).astype("string")
                pdf["domain_root"] = pdf.get("domain_root", pd.Series(pd.NA, index=pdf.index)).astype("string")

                # Build URL list - prioritize Google Maps websites if available
                url = pdf["siteweb"].fillna("").astype("string")
                fallback = pdf["domain_root"].fillna("").astype("string")
                
                if using_maps_data and "maps_websites" in pdf.columns:
                    maps_websites = pdf["maps_websites"].fillna("").astype("string")
                    # Parse multiple websites from maps (separated by '; ')
                    for idx, websites_str in maps_websites.items():
                        if websites_str and isinstance(websites_str, str):
                            websites = [w.strip() for w in websites_str.split(';') if w.strip()]
                            if websites and url.loc[idx] == "":
                                url.loc[idx] = websites[0]  # Use first website
                    
                    if logger:
                        maps_websites_found = (maps_websites != "").sum()
                        logger.debug(f"Found {maps_websites_found} websites from Google Maps data")

                url = url.where(url.str.strip().ne(""), fallback)

                mask = url.str.strip().ne("")
                url_ok = pd.Series(False, index=pdf.index)
                url_final = pd.Series("", index=pdf.index, dtype="string")

                url_map: Dict[int, str] = {
                    idx: str(raw).strip()
                    for idx, raw in url[mask].items()
                    if isinstance(raw, str) and raw.strip()
                }
                if not url_map:
                    pdf["url_site"] = url_final.astype("string")
                    pdf["http_ok"] = url_ok.astype("boolean")
                    table = pa.Table.from_pandas(pdf, preserve_index=False)
                    writer.write_table(table)
                    continue

                completed_extra = state.metadata.get("completed_extra", {}) if isinstance(state.metadata.get("completed_extra"), dict) else {}
                pending_keys = set(state.pending(url_map.values()))

                processed_map: Dict[int, str] = {}
                for idx, key in url_map.items():
                    if key in pending_keys:
                        processed_map[idx] = key
                    else:
                        record = completed_extra.get(key) if isinstance(completed_extra, dict) else {}
                        url_ok.at[idx] = bool(record.get("ok", True))
                        url_final.at[idx] = record.get("final_url", key)

                tasks = {}
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for idx, key in processed_map.items():
                        if key not in seen_urls:
                            total_known += 1
                            seen_urls.add(key)
                            state.set_metadata(total=total_known)
                        state.mark_started(key)
                        tasks[executor.submit(probe_cached, key)] = (idx, key)
                    for future in as_completed(tasks):
                        idx, key = tasks[future]
                        try:
                            ok, final = future.result()
                            url_ok.at[idx] = ok
                            url_final.at[idx] = final
                            state.mark_completed(
                                key,
                                extra={
                                    "ok": ok,
                                    "final_url": final,
                                },
                            )
                        except budget_middleware.BudgetExceededError:
                            raise
                        except Exception as exc:  # pragma: no cover - defensive
                            if logger:
                                logger.warning("probe failed for %s: %s", url.at[idx], exc)
                            url_ok.at[idx] = False
                            url_final.at[idx] = ""
                            state.mark_failed(key, str(exc))

                pdf["url_site"] = url_final.astype("string")
                pdf["http_ok"] = url_ok.astype("boolean")
                table = pa.Table.from_pandas(pdf, preserve_index=False)
                writer.write_table(table)
                total += len(pdf)

        state.set_metadata(
            last_output=str(outp),
            rows_processed=total,
            total=total_known,
        )

        return {
            "status": "OK", 
            "file": str(outp), 
            "rows": total, 
            "duration_s": round(time.time() - t0, 3),
            "using_maps_data": using_maps_data
        }
    except budget_middleware.BudgetExceededError:
        raise
    except Exception as exc:
        if logger:
            logger.exception("site probe failed: %s", exc)
        return {"status": "FAIL", "error": str(exc), "duration_s": round(time.time() - t0, 3)}
