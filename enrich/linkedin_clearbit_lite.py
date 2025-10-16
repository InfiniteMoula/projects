from __future__ import annotations

import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa

from utils.parquet import ArrowCsvWriter, ParquetBatchWriter, iter_batches

LOGGER = logging.getLogger("enrich.linkedin_clearbit_lite")

try:  # pragma: no cover - optional dependency
    import tldextract
except Exception:  # pragma: no cover - optional dependency
    tldextract = None


LEGAL_FORMS = {
    "sa",
    "sas",
    "sasu",
    "sarl",
    "sarlu",
    "eurl",
    "scop",
    "sci",
    "scca",
    "snc",
    "sca",
    "société",
    "societe",
    "association",
    "autoentrepreneur",
    "auto-entrepreneur",
    "ei",
    "eirl",
    "holding",
}

NAF_INDUSTRY_MAP: Dict[str, str] = {
    "01": "Farming",
    "02": "Farming",
    "10": "Food & Beverage",
    "11": "Food & Beverage",
    "14": "Textiles",
    "16": "Consumer Goods",
    "17": "Paper & Forest Products",
    "20": "Chemicals",
    "23": "Building Materials",
    "24": "Metals & Mining",
    "25": "Industrial Machinery",
    "26": "Information Technology & Services",
    "27": "Electrical & Electronic Manufacturing",
    "28": "Industrial Machinery",
    "29": "Automotive",
    "30": "Transportation & Logistics",
    "31": "Consumer Goods",
    "32": "Medical Devices",
    "33": "Industrial Services",
    "35": "Utilities",
    "36": "Utilities",
    "38": "Environmental Services",
    "41": "Construction",
    "42": "Construction",
    "43": "Construction",
    "45": "Automotive",
    "46": "Wholesale",
    "47": "Retail",
    "49": "Transportation & Logistics",
    "52": "Transportation & Logistics",
    "55": "Hospitality",
    "56": "Food & Beverage",
    "58": "Media & Publishing",
    "59": "Media & Entertainment",
    "60": "Media & Broadcasting",
    "61": "Telecommunications",
    "62": "Information Technology & Services",
    "63": "Information Technology & Services",
    "64": "Financial Services",
    "65": "Financial Services",
    "66": "Financial Services",
    "68": "Real Estate",
    "69": "Legal Services",
    "70": "Management Consulting",
    "71": "Architecture & Planning",
    "72": "Research",
    "73": "Marketing & Advertising",
    "74": "Professional Services",
    "75": "Veterinary",
    "77": "Leisure, Travel & Tourism",
    "78": "Staffing & Recruiting",
    "79": "Leisure, Travel & Tourism",
    "80": "Security & Investigations",
    "81": "Facilities Services",
    "82": "Business Supplies & Equipment",
    "84": "Government Administration",
    "85": "Education Management",
    "86": "Hospital & Health Care",
    "87": "Hospital & Health Care",
    "88": "Nonprofit Organization Management",
    "90": "Entertainment",
    "91": "Museums & Institutions",
    "92": "Gambling & Casinos",
    "93": "Sports & Recreation",
    "94": "Civic & Social Organization",
    "95": "Consumer Services",
    "96": "Consumer Services",
}

INDUSTRY_KEYWORDS: Sequence[Tuple[Sequence[str], str]] = (
    (("tech", "logiciel", "software", "digital", "data", "cloud"), "Information Technology & Services"),
    (("compta", "accounting"), "Accounting"),
    (("avocat", "legal", "law", "juridique"), "Legal Services"),
    (("marketing", "communication", "publicite", "media"), "Marketing & Advertising"),
    (("transport", "logistic", "livraison", "cargo"), "Transportation & Logistics"),
    (("hotel", "hospitality", "hostel", "restaurant", "resto", "bistro", "traiteur"), "Hospitality"),
    (("sante", "medical", "clinic", "hopital", "pharma", "paramedical"), "Hospital & Health Care"),
    (("immobilier", "realestate", "property"), "Real Estate"),
    (("btp", "batiment", "construction"), "Construction"),
    (("energie", "solar", "energie", "renew", "climat"), "Renewables & Environment"),
    (("ecole", "education", "formation", "campus", "lycee"), "Education Management"),
    (("banque", "finance", "assurance", "assureur"), "Financial Services"),
    (("beaute", "beauty", "esthetique", "cosmetic"), "Cosmetics"),
    (("agri", "ferme", "farm", "bio", "viticole", "viti"), "Farming"),
    (("artisan", "atelier", "menuiserie", "boulanger", "patisserie"), "Consumer Services"),
    (("event", "evenement", "festival"), "Events Services"),
    (("security", "securite", "gardiennage"), "Security & Investigations"),
)

EMPLOYEE_CODE_BUCKETS: Dict[str, str] = {
    "00": "1-10",
    "0": "1-10",
    "01": "1-10",
    "1": "1-10",
    "02": "1-10",
    "2": "1-10",
    "03": "1-10",
    "3": "1-10",
    "11": "11-50",
    "12": "11-50",
    "13": "11-50",
    "21": "51-200",
    "22": "51-200",
    "23": "51-200",
    "31": "201-500",
    "32": "201-500",
    "33": "201-500",
    "41": "501-1000",
    "42": "1001-5000",
    "43": "1001-5000",
    "51": "1001-5000",
    "52": "5001-10000",
    "53": "10000+",
    "NN": "1-10",
}

EMPLOYEE_KEYWORDS: Sequence[Tuple[Sequence[str], str]] = (
    (("startup", "studio", "atelier", "artisan", "boutique"), "1-10"),
    (("agence", "cabinet", "consult", "bureau"), "11-50"),
    (("group", "groupe", "industrie", "industrial"), "201-500"),
    (("international", "global", "holding"), "501-1000"),
)

DEFAULT_EMPLOYEE_RANGE = "1-10"


def _simplify(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _strip_legal_forms(slug: str) -> str:
    parts = [p for p in re.split(r"[-_/]+", slug) if p]
    filtered = [p for p in parts if p not in LEGAL_FORMS]
    return "-".join(filtered) if filtered else slug


def _slugify(text: str) -> str:
    simplified = _simplify(text)
    simplified = re.sub(r"[^a-z0-9]+", "-", simplified)
    simplified = simplified.strip("-")
    return _strip_legal_forms(simplified)


def _domain_from_value(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    if "@" in text and not text.startswith("http"):
        text = text.split("@", 1)[1]
    if not re.match(r"^[a-z][a-z0-9+.-]*://", text, flags=re.I):
        text = "https://" + text
    try:
        from urllib.parse import urlparse

        parsed = urlparse(text)
        host = parsed.netloc or parsed.path
    except Exception:
        host = text
    host = host.split("@")[-1]
    host = host.split(":")[0]
    host = host.strip().lower().strip(".")
    if not host:
        return ""
    host = re.sub(r"^www\d*\.", "", host)
    if tldextract:
        ext = tldextract.extract(host)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
    parts = [p for p in host.split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def _first_non_empty(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _extract_domain(row: Mapping[str, Any]) -> str:
    candidates = (
        "domain_root",
        "domain",
        "primary_domain",
        "website_domain",
        "url_site",
        "url_site_final",
        "url_siteweb",
        "site_web",
        "siteweb",
        "website",
        "url",
        "homepage",
        "site",
        "Nom site web",
        "Site web",
        "linkedin_url",
        "email",
        "Email générique",
    )
    for column in candidates:
        raw = row.get(column)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        domain = _domain_from_value(str(raw))
        if domain:
            return domain
    return ""


def _extract_naf(row: Mapping[str, Any]) -> Tuple[str, str]:
    naf_columns = (
        "naf",
        "naf_code",
        "naf_principal",
        "code_naf",
        "naf_apet",
        "nafape",
        "apet700",
        "activite_principale",
        "Secteur (NAF/APE)",
        "naf_libelle",
        "libelle_naf",
    )
    naf_label_columns = (
        "libelle_naf",
        "libellenaf",
        "activite_principale",
        "libelle_activite",
        "Secteur (NAF/APE)",
    )
    naf_code = ""
    for column in naf_columns:
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            naf_code = text
            break
    label = ""
    for column in naf_label_columns:
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            label = text
            break
    return naf_code, label


def _industry_from_naf(naf_code: str) -> str:
    if not naf_code:
        return ""
    digits = re.findall(r"\d", naf_code)
    if len(digits) < 2:
        return ""
    prefix = "".join(digits[:2])
    return NAF_INDUSTRY_MAP.get(prefix, "")


def _industry_from_keywords(text: str) -> str:
    simplified = _simplify(text)
    if not simplified:
        return ""
    for keywords, label in INDUSTRY_KEYWORDS:
        if any(keyword in simplified for keyword in keywords):
            return label
    return ""


def _guess_industry(row: Mapping[str, Any], domain: str) -> str:
    naf_code, naf_label = _extract_naf(row)
    industry = _industry_from_naf(naf_code)
    if industry:
        return industry
    industry = _industry_from_keywords(naf_label)
    if industry:
        return industry
    company_name = _first_non_empty(
        row,
        (
            "denomination",
            "denomination_usuelle",
            "raison_sociale",
            "enseigne",
            "Nom entreprise",
            "nom",
        ),
    )
    industry = _industry_from_keywords(company_name)
    if industry:
        return industry
    industry = _industry_from_keywords(domain)
    if industry:
        return industry
    return ""


def _bucket_employee_count(value: int) -> str:
    if value <= 10:
        return "1-10"
    if value <= 50:
        return "11-50"
    if value <= 200:
        return "51-200"
    if value <= 500:
        return "201-500"
    if value <= 1000:
        return "501-1000"
    if value <= 5000:
        return "1001-5000"
    if value <= 10000:
        return "5001-10000"
    return "10000+"


def _normalize_employee_value(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (float, int)) and pd.isna(raw):
        return ""
    try:
        if pd.isna(raw):
            return ""
    except Exception:
        pass
    if isinstance(raw, str) and not raw.strip():
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    cleaned = text.replace("\xa0", " ")
    code = cleaned.upper().replace(" ", "")
    if code in EMPLOYEE_CODE_BUCKETS:
        return EMPLOYEE_CODE_BUCKETS[code]
    digits = [int(match) for match in re.findall(r"\d+", cleaned)]
    if digits:
        approx = max(digits)
        return _bucket_employee_count(approx)
    lowered = cleaned.lower()
    for keywords, label in EMPLOYEE_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return label
    if "plus" in lowered or "+" in lowered or ">" in lowered:
        return "10000+"
    if "aucun" in lowered or "0" == lowered:
        return "1-10"
    return ""


def _guess_employee_range(row: Mapping[str, Any], domain: str, default: str = DEFAULT_EMPLOYEE_RANGE) -> str:
    employee_columns = (
        "employee_range",
        "Effectif",
        "effectif",
        "effectif_salarie",
        "effectifs",
        "tranche_effectif",
        "trancheEffectifsEtablissement",
        "trancheEffectifsUniteLegale",
        "tranche_effectifs",
        "nb_salaries",
        "nombre_salaries",
    )
    for column in employee_columns:
        value = row.get(column)
        normalized = _normalize_employee_value(value)
        if normalized:
            return normalized
    domain_text = domain.replace("-", "") if isinstance(domain, str) else ""
    for keywords, label in EMPLOYEE_KEYWORDS:
        if any(keyword in domain_text for keyword in keywords):
            return label
    return default


def _build_linkedin_url(row: Mapping[str, Any], domain: str, cfg: Mapping[str, Any]) -> str:
    if not isinstance(cfg, Mapping):
        cfg = {}
    base_url = str(cfg.get("base_url") or "https://www.linkedin.com/company/").rstrip("/") + "/"
    slug_candidate = ""
    if domain:
        slug_candidate = domain.split(".")[0]
    if not slug_candidate:
        slug_candidate = _first_non_empty(
            row,
            (
                "denomination",
                "denomination_usuelle",
                "raison_sociale",
                "enseigne",
                "Nom entreprise",
                "nom",
            ),
        )
    slug = _slugify(slug_candidate)
    if not slug:
        return ""
    return f"{base_url}{slug}/"


def _ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "industry" not in work.columns:
        work["industry"] = pd.Series(pd.NA, index=work.index, dtype="string")
    if "employee_range" not in work.columns:
        work["employee_range"] = pd.Series(pd.NA, index=work.index, dtype="string")
    if "linkedin_url" not in work.columns:
        work["linkedin_url"] = pd.Series(pd.NA, index=work.index, dtype="string")
    if "has_domain" not in work.columns:
        work["has_domain"] = pd.Series(pd.NA, index=work.index, dtype="boolean")
    return work


def process_linkedin_clearbit_lite(
    df_in: pd.DataFrame,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = cfg or {}
    df_out = _ensure_columns(df_in if df_in is not None else pd.DataFrame())
    if df_out.empty:
        df_out["industry"] = df_out["industry"].astype("string")
        df_out["employee_range"] = df_out["employee_range"].astype("string")
        df_out["linkedin_url"] = df_out["linkedin_url"].astype("string")
        df_out["has_domain"] = df_out["has_domain"].astype("boolean")
        return df_out, {"rows": 0, "with_domain": 0, "linkedin_urls": 0}

    industries: Dict[int, str] = {}
    employee_ranges: Dict[int, str] = {}
    linkedin_urls: Dict[int, str] = {}
    has_domain: Dict[int, bool] = {}

    existing_industry = df_out["industry"].astype("string")
    existing_employees = df_out["employee_range"].astype("string")
    existing_linkedin = df_out["linkedin_url"].astype("string")

    default_range = str(cfg.get("default_employee_range") or DEFAULT_EMPLOYEE_RANGE)

    for idx, row in df_out.iterrows():
        domain = _extract_domain(row)
        has_domain[idx] = bool(domain)

        industry = _guess_industry(row, domain)
        existing_industry_value = existing_industry.get(idx, pd.NA)
        has_existing_industry = False
        if pd.notna(existing_industry_value):
            has_existing_industry = bool(str(existing_industry_value).strip())
        if industry and not has_existing_industry:
            industries[idx] = industry

        employee_range = _guess_employee_range(row, domain, default=default_range)
        existing_employee_value = existing_employees.get(idx, pd.NA)
        has_existing_employee = False
        if pd.notna(existing_employee_value):
            has_existing_employee = bool(str(existing_employee_value).strip())
        if employee_range and not has_existing_employee:
            employee_ranges[idx] = employee_range

        linkedin_url = _build_linkedin_url(row, domain, cfg)
        existing_linkedin_value = existing_linkedin.get(idx, pd.NA)
        has_existing_linkedin = False
        if pd.notna(existing_linkedin_value):
            has_existing_linkedin = bool(str(existing_linkedin_value).strip())
        if linkedin_url and not has_existing_linkedin:
            linkedin_urls[idx] = linkedin_url

    df_out["industry"] = existing_industry
    df_out["employee_range"] = existing_employees
    df_out["linkedin_url"] = existing_linkedin

    if industries:
        update = pd.Series(industries, dtype="string")
        df_out.loc[update.index, "industry"] = update
    if employee_ranges:
        update = pd.Series(employee_ranges, dtype="string")
        df_out.loc[update.index, "employee_range"] = update
    if linkedin_urls:
        update = pd.Series(linkedin_urls, dtype="string")
        df_out.loc[update.index, "linkedin_url"] = update

    df_out.loc[:, "has_domain"] = pd.Series(has_domain).reindex(df_out.index).astype("boolean")

    linkedin_count = int(df_out["linkedin_url"].fillna("").astype("string").str.strip().ne("").sum())
    domain_count = int(df_out["has_domain"].fillna(False).astype("boolean").sum())
    industry_count = int(df_out["industry"].fillna("").astype("string").str.strip().ne("").sum())
    employee_count = int(df_out["employee_range"].fillna("").astype("string").str.strip().ne("").sum())

    summary = {
        "rows": len(df_out),
        "with_domain": domain_count,
        "linkedin_urls": linkedin_count,
        "industry_filled": industry_count,
        "employee_range_filled": employee_count,
    }
    return df_out, summary


def _load_batches(path: Path) -> Iterable[pd.DataFrame]:
    if path.suffix.lower() == ".parquet":
        yield from iter_batches(path)
    else:
        df = pd.read_csv(path)
        yield df


def run(cfg: Mapping[str, Any] | None, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    start = time.time()
    context_logger = ctx.get("logger") if isinstance(ctx, Mapping) else None
    logger = context_logger or LOGGER
    log_enabled = context_logger is not None
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))

    input_candidates = [
        outdir / "domains_enriched.parquet",
        outdir / "enriched_site.parquet",
        outdir / "enriched_domain.parquet",
        outdir / "normalized.parquet",
        outdir / "normalized.csv",
    ]
    source_path = next((candidate for candidate in input_candidates if candidate.exists()), None)
    if source_path is None:
        if log_enabled:
            logger.warning("linkedin_clearbit_lite skipped: no dataset available")
        return {"status": "SKIPPED", "reason": "NO_INPUT_DATA"}

    parquet_path = outdir / "linkedin_clearbit_lite.parquet"
    csv_path = outdir / "linkedin_clearbit_lite.csv"
    for target in (parquet_path, csv_path):
        if target.exists():
            try:
                target.unlink()
            except OSError:
                pass

    total_rows = 0
    domain_rows = 0
    linkedin_rows = 0
    industry_rows = 0
    employee_rows = 0

    try:
        with ParquetBatchWriter(parquet_path) as parquet_writer, ArrowCsvWriter(csv_path) as csv_writer:
            for batch in _load_batches(source_path):
                if batch.empty:
                    continue
                processed, summary = process_linkedin_clearbit_lite(batch, cfg)
                table = pa.Table.from_pandas(processed, preserve_index=False)
                parquet_writer.write_table(table)
                csv_writer.write_table(table)

                total_rows += summary.get("rows", len(processed))
                domain_rows += summary.get("with_domain", 0)
                linkedin_rows += summary.get("linkedin_urls", 0)
                industry_rows += summary.get("industry_filled", 0)
                employee_rows += summary.get("employee_range_filled", 0)
    except Exception as exc:  # pragma: no cover - defensive
        if log_enabled:
            logger.exception("linkedin_clearbit_lite failed: %s", exc)
        return {"status": "FAIL", "error": str(exc)}

    duration = round(time.time() - start, 3)
    if log_enabled:
        logger.info(
            "linkedin_clearbit_lite completed: rows=%s domains=%s linkedin_urls=%s",
            total_rows,
            domain_rows,
            linkedin_rows,
        )

    return {
        "status": "OK",
        "file": str(parquet_path),
        "rows": total_rows,
        "with_domain": domain_rows,
        "linkedin_urls": linkedin_rows,
        "industry_filled": industry_rows,
        "employee_range_filled": employee_rows,
        "duration_s": duration,
        "source": str(source_path),
    }


__all__ = ["process_linkedin_clearbit_lite", "run"]
