
# FILE: enrich/dns_checks.py
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import dns.resolver
except Exception:  # pragma: no cover - optional dependency
    dns = None

from utils.parquet import ParquetBatchWriter, iter_batches

DNS_TIMEOUT = 2.5
DEFAULT_CACHE_TTL = 900


def _bool(s: pd.Series) -> pd.Series:
    return s.astype("boolean")


def run(cfg, ctx):
    t0 = time.time()
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    
    # Primary input: Google Maps enriched data (to be mostly useless as requested)
    maps_inp = outdir / "google_maps_enriched.parquet"
    # Fallback input: domain enriched data
    domain_inp = outdir / "enriched_domain.parquet"
    
    src = maps_inp if maps_inp.exists() else domain_inp
    outp = outdir / "enriched_dns.parquet"
    if not src.exists():
        return {"status": "WARN", "error": "no input for dns checks"}

    ttl = int(ctx.get("dns_cache_ttl", DEFAULT_CACHE_TTL))
    workers = max(1, int(ctx.get("workers", 4)))
    logger = ctx.get("logger")
    using_maps_data = src == maps_inp
    
    # As requested - make DNS checks mostly useless when using Google Maps data
    if using_maps_data and logger:
        logger.info("DNS checks using Google Maps data - minimal functionality as requested")

    pf = pq.ParquetFile(str(src))
    base_schema = pf.schema_arrow
    fields = list(base_schema)
    if "dns_ok" not in pf.schema.names:
        fields.append(pa.field("dns_ok", pa.bool_()))
    if "mx_ok" not in pf.schema.names:
        fields.append(pa.field("mx_ok", pa.bool_()))
    # Create schema that includes all existing fields plus dns/mx fields
    schema = pa.schema(fields)

    resolver = None
    if dns and not using_maps_data:  # Skip DNS resolution when using maps data
        resolver = dns.resolver.Resolver(configure=True)
        resolver.lifetime = DNS_TIMEOUT
        resolver.timeout = DNS_TIMEOUT

    cache: Dict[Tuple[str, str], Tuple[float, bool]] = {}

    def resolve_many(domains, record_type: str) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        now = time.time()
        pending = []
        
        # If using maps data, assume all domains are valid (making DNS checks useless)
        if using_maps_data:
            for domain in domains:
                results[domain] = True  # Assume valid since it came from Google Maps
                cache[(record_type, domain)] = (now + ttl, True)
            return results
        
        for domain in domains:
            key = (record_type, domain)
            cached = cache.get(key)
            if cached and cached[0] > now:
                results[domain] = cached[1]
            else:
                pending.append(domain)

        if not pending:
            return results

        if resolver is None:
            for domain in pending:
                results[domain] = False
                cache[(record_type, domain)] = (now + ttl, False)
            return results

        def query(domain: str) -> bool:
            try:
                if not domain:
                    return False
                resolver.resolve(domain, record_type)
                return True
            except Exception:
                return False

        with ThreadPoolExecutor(max_workers=min(workers, len(pending)) or 1) as executor:
            future_map = {executor.submit(query, domain): domain for domain in pending}
            for future in future_map:
                domain = future_map[future]
                try:
                    value = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    if logger:
                        logger.warning("dns query failed for %s %s: %s", domain, record_type, exc)
                    value = False
                results[domain] = value
                cache[(record_type, domain)] = (now + ttl, value)
        return results

    total = 0

    try:
        with ParquetBatchWriter(outp) as writer:
            for pdf in iter_batches(src):
                if pdf.empty:
                    continue
                if "domain_root" not in pdf.columns:
                    pdf["domain_root"] = pd.NA
                domains = pdf["domain_root"].astype("string").fillna("").str.strip()
                unique_domains = sorted(set(domains.tolist()))

                a_results = resolve_many(unique_domains, "A")
                mx_results = resolve_many(unique_domains, "MX")

                dns_ok = domains.map(lambda d: a_results.get(d, False))
                mx_ok = domains.map(lambda d: mx_results.get(d, False))

                pdf["dns_ok"] = _bool(dns_ok)
                pdf["mx_ok"] = _bool(mx_ok)

                table = pa.Table.from_pandas(pdf, preserve_index=False)
                writer.write_table(table)
                total += len(pdf)

        return {"status": "OK", "file": str(outp), "rows": total, "duration_s": round(time.time() - t0, 3)}
    except Exception as exc:
        if logger:
            logger.exception("dns checks failed: %s", exc)
        return {"status": "FAIL", "error": str(exc), "duration_s": round(time.time() - t0, 3)}
