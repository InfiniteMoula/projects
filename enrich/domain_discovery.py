
# FILE: enrich/domain_discovery.py
import re
import time
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow as pa

try:
    import tldextract
except Exception:  # pragma: no cover - optional dependency
    tldextract = None

from ml.domain_features import coerce_candidates, normalize_domain
from ml.domain_predictor import predict_best_domain

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


def _first_value(row: pd.Series, columns: List[str]) -> str:
    for column in columns:
        if column not in row:
            continue
        value = row[column]
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        elif pd.notna(value):
            text = str(value).strip()
            if text:
                return text
    return ""


def _split_emails(raw: object) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [item.strip() for item in re.split(r"[;,\s]+", raw) if item.strip()]
        return [item for item in items if "@" in item]
    if isinstance(raw, (list, tuple, set)):
        emails: List[str] = []
        for item in raw:
            if not item:
                continue
            text = str(item).strip()
            if "@" in text:
                emails.append(text)
        return emails
    return []


def _collect_domain_candidates(row: pd.Series) -> List[str]:
    candidates: List[str] = []
    for column in ("domain_candidates", "site_web_candidates", "website_candidates", "serp_candidates"):
        if column in row and row[column] is not None:
            extracted = coerce_candidates(row[column])
            for candidate in extracted:
                domain = ""
                if isinstance(candidate, dict):
                    for key in ("domain", "url", "site", "homepage", "value", "candidate", "website"):
                        if key in candidate:
                            domain = normalize_domain(candidate[key])
                            if domain:
                                break
                if not domain:
                    domain = normalize_domain(candidate)
                if not domain:
                    domain = _domain_from_url(str(candidate))
                if domain and domain not in candidates:
                    candidates.append(domain)
    if not candidates:
        from_site = _domain_from_url(str(row.get("siteweb")))
        if from_site and from_site not in candidates:
            candidates.append(from_site)
    for email in _split_emails(row.get("email")) + _split_emails(row.get("maps_emails")):
        email_domain = _domain_from_email(email)
        if email_domain and email_domain not in candidates:
            candidates.append(email_domain)
    return candidates


# --- main --------------------------------------------------------------------
def run(cfg: dict, ctx: dict) -> dict:
    t0 = time.time()
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    
    # Primary input: Google Maps enriched data  
    maps_inp = outdir / "google_maps_enriched.parquet"
    # Fallback input: normalized data
    normalized_inp = outdir / "normalized.parquet"
    
    inp = maps_inp if maps_inp.exists() else normalized_inp
    outp = outdir / "enriched_domain.parquet"

    if not inp.exists():
        return {"status": "FAIL", "error": f"missing input: {inp}"}

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
        pa.field("predicted_domain", pa.string()),
        pa.field("predicted_domain_proba", pa.float64()),
    ])

    wanted = [f.name for f in schema]
    total = 0
    logger = ctx.get("logger")
    using_maps_data = inp == maps_inp

    if logger:
        logger.info(f"Domain discovery using {'Google Maps' if using_maps_data else 'normalized'} data")

    try:
        with ParquetBatchWriter(outp, schema=schema) as writer:
            for pdf in iter_batches(inp, columns=None):  # Read all columns
                if pdf.empty:
                    continue

                # Ensure required columns exist
                float_columns = {"predicted_domain_proba"}
                string_columns = [c for c in wanted if c not in float_columns]

                for column in string_columns:
                    if column not in pdf.columns:
                        pdf[column] = pd.NA
                    pdf[column] = _as_str(pdf[column])

                for column in float_columns:
                    if column not in pdf.columns:
                        pdf[column] = pd.Series(pd.array([pd.NA] * len(pdf), dtype="Float64"))
                    else:
                        numeric = pd.to_numeric(pdf[column], errors="coerce")
                        pdf[column] = pd.Series(pd.array(numeric, dtype="Float64"))

                # Extract domains - prioritize Google Maps data if available
                if using_maps_data:
                    # Use Google Maps websites and emails first
                    maps_websites = _strip(pdf.get("maps_websites", pd.Series("", index=pdf.index)))
                    maps_emails = _strip(pdf.get("maps_emails", pd.Series("", index=pdf.index)))
                    
                    # Extract domains from Google Maps data
                    maps_website_domains = maps_websites.map(_domain_from_url)
                    maps_email_domains = maps_emails.map(_domain_from_email)
                    
                    # Use Google Maps domains if available, fallback to original data
                    site = _strip(pdf["siteweb"])
                    url_domain = site.map(_domain_from_url)
                    email_dom = _strip(pdf["email"]).map(_domain_from_email)
                    
                    # Priority: Maps websites -> Maps emails -> Original websites -> Original emails
                    pdf["domain_root"] = (
                        maps_website_domains
                        .where(maps_website_domains != "", maps_email_domains)
                        .where(maps_email_domains != "", url_domain)
                        .where(url_domain != "", email_dom)
                    )
                    
                    if logger:
                        maps_domains_found = (maps_website_domains != "").sum() + (maps_email_domains != "").sum()
                        logger.debug(f"Found {maps_domains_found} domains from Google Maps data")
                else:
                    # Original logic when Google Maps data not available
                    site = _strip(pdf["siteweb"])
                    url_domain = site.map(_domain_from_url)
                    email_dom = _strip(pdf["email"]).map(_domain_from_email)
                    pdf["domain_root"] = url_domain.where(url_domain != "", email_dom)

                pdf["predicted_domain"] = pd.Series(pd.array([pd.NA] * len(pdf), dtype="string"))
                pdf["predicted_domain_proba"] = pd.Series(pd.array([pd.NA] * len(pdf), dtype="Float64"))

                missing_mask = _as_str(pdf["domain_root"]).fillna("").str.strip() == ""
                if missing_mask.any():
                    for idx in pdf.index[missing_mask]:
                        row = pdf.loc[idx]
                        candidates = _collect_domain_candidates(row)
                        if not candidates:
                            continue
                        name = _first_value(row, ["raison_sociale", "enseigne", "nom"])
                        naf_code = str(row.get("naf") or "")
                        city = _first_value(row, ["commune", "ville"])
                        postal_code = _first_value(row, ["cp", "code_postal"])
                        emails = _split_emails(row.get("email")) + _split_emails(row.get("maps_emails"))
                        unique_emails = list(dict.fromkeys(emails))
                        predicted, proba = predict_best_domain(
                            name=name,
                            naf=naf_code,
                            city=city,
                            candidates=candidates,
                            postal_code=postal_code or None,
                            emails=unique_emails or None,
                        )
                        if not predicted or proba <= 0:
                            continue
                        pdf.at[idx, "predicted_domain"] = predicted
                        pdf.at[idx, "predicted_domain_proba"] = round(float(proba), 4)
                        current_value = str(pdf.at[idx, "domain_root"] or "").strip().lower()
                        if not current_value:
                            pdf.at[idx, "domain_root"] = predicted

                table = pa.Table.from_pandas(pdf[wanted], preserve_index=False, schema=schema)
                writer.write_table(table)
                total += len(pdf)

        return {
            "status": "OK", 
            "file": str(outp), 
            "rows": total, 
            "duration_s": round(time.time() - t0, 3),
            "using_maps_data": using_maps_data
        }
    except Exception as exc:
        if logger:
            logger.exception("domain discovery failed: %s", exc)
        return {"status": "FAIL", "error": str(exc), "duration_s": round(time.time() - t0, 3)}
