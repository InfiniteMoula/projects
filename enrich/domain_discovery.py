
# FILE: enrich/domain_discovery.py
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa

try:
    import tldextract
except Exception:  # pragma: no cover - optional dependency
    tldextract = None

from utils.parquet import ParquetBatchWriter, iter_batches

# --- helpers -----------------------------------------------------------------
def _as_str(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(pd.array([], dtype="string"), dtype="string")
    return s.astype("string")


def _strip(s: pd.Series) -> pd.Series:
    return _as_str(s).fillna("").str.strip()


def _domain_from_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    u = url.strip()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    if tldextract:
        ext = tldextract.extract(u)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
    m = re.search(r"^https?://([^/]+)", u, flags=re.I)
    host = (m.group(1) if m else u).lower()
    host = host.split("@")[-1]
    host = host.split(":")[0]
    parts = [p for p in host.split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def _domain_from_email(email: str) -> str:
    if not isinstance(email, str):
        return ""
    email = email.strip()
    if "@" in email:
        return email.split("@", 1)[1].lower()
    return ""


# --- main --------------------------------------------------------------------
def run(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    inp = outdir / "normalized.parquet"
    outp = outdir / "enriched_domain.parquet"

    if not inp.exists():
        return {"status": "FAIL", "error": f"missing {inp}"}

    schema = pa.schema([
        pa.field("siren", pa.string()),
        pa.field("siret", pa.string()),
        pa.field("raison_sociale", pa.string()),
        pa.field("enseigne", pa.string()),
        pa.field("commune", pa.string()),
        pa.field("cp", pa.string()),
        pa.field("adresse", pa.string()),
        pa.field("naf", pa.string()),
        pa.field("date_creation", pa.string()),
        pa.field("telephone_norm", pa.string()),
        pa.field("email", pa.string()),
        pa.field("siteweb", pa.string()),
        pa.field("nom", pa.string()),
        pa.field("prenom", pa.string()),
        pa.field("domain_root", pa.string()),
    ])

    wanted = [f.name for f in schema]
    total = 0
    logger = ctx.get("logger")

    try:
        with ParquetBatchWriter(outp, schema=schema) as writer:
            for pdf in iter_batches(inp, columns=wanted):
                if pdf.empty:
                    continue

                for c in wanted:
                    if c not in pdf.columns:
                        pdf[c] = pd.NA
                    pdf[c] = _as_str(pdf[c])

                site = _strip(pdf["siteweb"])
                url_domain = site.map(_domain_from_url)
                email_dom = _strip(pdf["email"]).map(_domain_from_email)
                pdf["domain_root"] = url_domain.where(url_domain != "", email_dom)

                table = pa.Table.from_pandas(pdf[wanted], preserve_index=False, schema=schema)
                writer.write_table(table)
                total += len(pdf)

        return {"status": "OK", "file": str(outp), "rows": total, "duration_s": round(time.time() - t0, 3)}
    except Exception as exc:
        if logger:
            logger.exception("domain discovery failed: %s", exc)
        return {"status": "FAIL", "error": str(exc), "duration_s": round(time.time() - t0, 3)}
