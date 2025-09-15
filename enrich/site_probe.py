
# FILE: enrich/site_probe.py
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pyarrow as pa
import requests

from utils.parquet import ParquetBatchWriter, iter_batches

REQ_TIMEOUT = 4.0
UA = "Mozilla/5.0 (+contact-probe)"
DEFAULT_CACHE_TTL = 900
DEFAULT_RETRIES = 2


def _probe(url: str, *, timeout: float) -> tuple[bool, str]:
    if not isinstance(url, str) or not url.strip():
        return False, ""
    u = url.strip()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    try:
        response = requests.head(
            u,
            allow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": UA},
        )
        ok = 200 <= response.status_code < 400
        return ok, response.url if ok else ""
    except Exception:
        return False, ""


def _probe_with_retry(url: str, *, timeout: float, retries: int) -> tuple[bool, str]:
    last: tuple[bool, str] = (False, "")
    for _ in range(max(retries, 1)):
        last = _probe(url, timeout=timeout)
        if last[0]:
            break
    return last


def run(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    inp = outdir / "enriched_domain.parquet"
    outp = outdir / "enriched_site.parquet"
    if not inp.exists():
        return {"status": "WARN", "error": f"missing {inp}"}

    timeout = float(ctx.get("site_probe_timeout", REQ_TIMEOUT))
    ttl = int(ctx.get("site_probe_ttl", DEFAULT_CACHE_TTL))
    retries = int(ctx.get("site_probe_retries", DEFAULT_RETRIES))
    workers = max(1, int(ctx.get("workers", 4)))
    dry_run = bool(ctx.get("dry_run", False))
    logger = ctx.get("logger")

    cache: Dict[str, Tuple[float, tuple[bool, str]]] = {}

    def probe_cached(raw_url: str) -> tuple[bool, str]:
        if dry_run:
            return False, ""
        key = raw_url.strip().lower()
        now = time.time()
        result = cache.get(key)
        if result and result[0] > now:
            return result[1]
        value = _probe_with_retry(raw_url, timeout=timeout, retries=retries)
        cache[key] = (now + ttl, value)
        return value

    total = 0

    try:
        with ParquetBatchWriter(outp) as writer:
            for pdf in iter_batches(inp):
                if pdf.empty:
                    continue

                pdf["siteweb"] = pdf.get("siteweb", pd.Series(pd.NA, index=pdf.index)).astype("string")
                pdf["domain_root"] = pdf.get("domain_root", pd.Series(pd.NA, index=pdf.index)).astype("string")

                url = pdf["siteweb"].fillna("").astype("string")
                fallback = pdf["domain_root"].fillna("").astype("string")
                url = url.where(url.str.strip().ne(""), fallback)

                mask = url.str.strip().ne("")
                url_ok = pd.Series(False, index=pdf.index)
                url_final = pd.Series("", index=pdf.index, dtype="string")

                tasks = {}
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for idx, raw in url[mask].items():
                        tasks[executor.submit(probe_cached, str(raw))] = idx
                    for future in as_completed(tasks):
                        idx = tasks[future]
                        try:
                            ok, final = future.result()
                        except Exception as exc:  # pragma: no cover - defensive
                            if logger:
                                logger.warning("probe failed for %s: %s", url.at[idx], exc)
                            ok, final = False, ""
                        url_ok.at[idx] = ok
                        url_final.at[idx] = final

                pdf["url_site"] = url_final.astype("string")
                pdf["http_ok"] = url_ok.astype("boolean")
                table = pa.Table.from_pandas(pdf, preserve_index=False)
                writer.write_table(table)
                total += len(pdf)

        return {"status": "OK", "file": str(outp), "rows": total, "duration_s": round(time.time() - t0, 3)}
    except Exception as exc:
        if logger:
            logger.exception("site probe failed: %s", exc)
        return {"status": "FAIL", "error": str(exc), "duration_s": round(time.time() - t0, 3)}
