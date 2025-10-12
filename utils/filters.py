"""Filtering utilities for premium, CRM-ready datasets."""
from __future__ import annotations

import ast
import logging
import math
import re
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse, urlunparse

import pandas as pd

LOGGER = logging.getLogger("utils.filters")

PREMIUM_COLUMNS: list[str] = [
    "company_name",
    "domain",
    "site_web",
    "siren",
    "siret",
    "adresse_complete",
    "code_postal",
    "ville",
    "departement",
    "pays",
    "naf_code",
    "naf",
    "employee_count",
    "revenue_range",
    "date_creation",
    "linkedin_company_url",
    "tech_stack",
    "email",
    "telephone",
    "etat_administratif",
    "score_quality",
]

_EMPLOYEE_BUCKETS: list[tuple[int, Optional[int], str]] = [
    (1, 10, "1-10"),
    (11, 50, "11-50"),
    (51, 200, "51-200"),
    (201, 1000, "201-1000"),
    (1001, None, "1000+"),
]

_TRUE_VALUES = {"true", "1", "yes", "y", "t"}


def _clean_text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
    return str(value).strip()


def _series_from_columns(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    available = [col for col in candidates if col in df.columns]
    if not available:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    subset = df[available].copy()
    for column in available:
        subset[column] = subset[column].map(_clean_text_value)
        subset[column] = subset[column].replace("", pd.NA)
    combined = subset.bfill(axis=1).iloc[:, 0]
    combined = combined.fillna("")
    return combined


def _normalize_domain(value: str) -> Optional[str]:
    text = _clean_text_value(value)
    if not text:
        return None
    if "@" in text:
        return None
    if not re.match(r"^[a-z]+://", text, flags=re.I):
        text = f"https://{text}"
    parsed = urlparse(text)
    host = parsed.netloc or parsed.path
    host = host.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    # Remove port
    host = host.split("@")[-1]
    host = host.split(":")[0]
    if not host:
        return None
    return host


def _normalize_url(value: str, *, default_domain: Optional[str] = None) -> Optional[str]:
    text = _clean_text_value(value)
    if not text and default_domain:
        text = default_domain
    if not text:
        return None
    if not re.match(r"^[a-z]+://", text, flags=re.I):
        text = f"https://{text}"
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        scheme = "https"
    netloc = parsed.netloc or parsed.path
    if not netloc and default_domain:
        netloc = default_domain
    if not netloc:
        return None
    netloc = netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path or ""
    if path in ("/", ""):
        path = ""
    query_pairs = []
    if parsed.query:
        for item in parsed.query.split("&"):
            if not item:
                continue
            if item.lower().startswith("utm_"):
                continue
            query_pairs.append(item)
    query = "&".join(query_pairs)
    normalized = urlunparse((scheme, netloc, path, "", query, ""))
    if normalized.endswith("//"):
        normalized = normalized[:-1]
    return normalized


def _normalize_linkedin(value: str) -> Optional[str]:
    url = _normalize_url(value)
    if not url:
        return None
    parsed = urlparse(url)
    if "linkedin.com" not in parsed.netloc:
        return None
    path = parsed.path or ""
    match = re.match(r"/company/([^/]+)/?", path, flags=re.I)
    if match:
        slug = match.group(1).strip()
        if slug:
            return f"https://www.linkedin.com/company/{slug}/"
    return None


def _extract_first_token(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            token = _clean_text_value(item)
            if token:
                return token
        return None
    text = _clean_text_value(value)
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return _extract_first_token(list(parsed))
        except (ValueError, SyntaxError):
            pass
    for separator in (";", ",", "|", " "):
        if separator in text:
            parts = [part.strip() for part in text.split(separator)]
            for part in parts:
                if part:
                    return part
    return text


def _normalize_employee_bucket(value: object) -> Optional[str]:
    text = _clean_text_value(value)
    if not text:
        return None
    numbers = [int(match.group()) for match in re.finditer(r"\d+", text)]
    if numbers:
        approx = max(numbers)
        if approx == 0:
            return "1-10"
        for lower, upper, label in _EMPLOYEE_BUCKETS:
            if upper is None and approx >= lower:
                return label
            if lower <= approx <= (upper or approx):
                return label
        return "1000+"
    normalized = text.lower()
    if "aut" in normalized or "n/a" in normalized:
        return None
    mapping = {
        "micro": "1-10",
        "petite": "11-50",
        "moyenne": "51-200",
        "grande": "201-1000",
    }
    for key, bucket in mapping.items():
        if key in normalized:
            return bucket
    return None


def _normalize_boolean(value: object) -> Optional[bool]:
    text = _clean_text_value(value).lower()
    if not text:
        return None
    if text in _TRUE_VALUES:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _normalize_tech_stack(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        tokens = [_clean_text_value(item) for item in value if _clean_text_value(item)]
    else:
        text = _clean_text_value(value)
        if not text:
            return None
        parsed = None
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, (list, tuple, set)):
            tokens = [_clean_text_value(item) for item in parsed if _clean_text_value(item)]
        else:
            tokens = [
                item.strip()
                for sep in (";", ",", "|")
                for item in text.split(sep)
                if sep in text
            ]
            if not tokens:
                tokens = [text]
            tokens = [_clean_text_value(item) for item in tokens if _clean_text_value(item)]
    if not tokens:
        return None
    unique: list[str] = []
    seen = set()
    for token in tokens:
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(token)
        if len(unique) >= 8:
            break
    return ", ".join(unique)


def filter_premium_columns(df: pd.DataFrame, *, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=PREMIUM_COLUMNS)

    work = df.copy()
    log = logger or LOGGER

    work["company_name"] = _series_from_columns(
        work,
        [
            "company_name",
            "denomination_usuelle",
            "enseigne",
            "name",
            "raison_sociale",
            "denomination",
        ],
    ).str.title()

    city_series = _series_from_columns(
        work,
        [
            "ville",
            "commune",
            "libelle_commune",
            "libellecommuneetablissement",
            "libellecommune2etablissement",
        ],
    )

    code_postal_series = _series_from_columns(
        work,
        [
            "code_postal",
            "cp",
            "codepostal2etablissement",
            "code_postal2",
            "codepostal",
        ],
    )

    num_voie_series = _series_from_columns(
        work,
        [
            "numero_voie",
            "numerovoie2etablissement",
            "numerovoie",
            "numerodernieretablissement",
        ],
    )

    libelle_voie_series = _series_from_columns(
        work,
        [
            "libelle_voie",
            "libellevoie2etablissement",
            "libellevoie",
            "type_voie",
        ],
    )

    line1 = (
        num_voie_series.str.cat(libelle_voie_series, sep=" ", na_rep="")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace({"nan": ""})
    )
    line2 = (
        code_postal_series.str.cat(city_series, sep=" ", na_rep="")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace({"nan": ""})
    )

    addr_values: list[str] = []
    for value1, value2 in zip(line1.tolist(), line2.tolist()):
        parts = []
        if value1:
            parts.append(value1)
        if value2:
            parts.append(value2)
        addr_values.append(", ".join(parts))
    adresse_complete = pd.Series(addr_values, index=work.index)
    work["adresse_complete"] = adresse_complete.replace("", pd.NA)

    work["code_postal"] = code_postal_series.replace("", pd.NA)
    work["ville"] = city_series.replace("", pd.NA)

    work["departement"] = _series_from_columns(
        work,
        ["departement", "department", "code_departement", "departementetablissement"],
    ).replace("", pd.NA)

    pays_series = _series_from_columns(
        work,
        [
            "pays",
            "libellepaysetrangeretablissement",
            "libellepaysetranger2etablissement",
            "pays_iso",
        ],
    )
    if "code_postal" in work.columns and pays_series.eq("").all():
        pays_series = pd.Series(["France"] * len(work), index=work.index)
    work["pays"] = pays_series.replace("", pd.NA)

    domain_candidates = _series_from_columns(
        work,
        ["domain", "site_web", "website", "site", "url", "homepage", "contact_domain"],
    )
    domain_normalized = domain_candidates.map(_normalize_domain)
    work["domain"] = domain_normalized

    site_candidates = _series_from_columns(
        work,
        ["site_web", "website", "site", "url", "homepage"],
    )
    work["site_web"] = site_candidates.combine(
        domain_normalized, lambda site_value, dom: _normalize_url(site_value, default_domain=dom)
    )
    work["site_web"] = work["site_web"].fillna(
        domain_normalized.map(lambda domain_value: f"https://{domain_value}" if domain_value else None)
    )

    work["linkedin_company_url"] = _series_from_columns(
        work,
        ["linkedin_company_url", "linkedin", "linkedin_url", "linkedin_company"],
    ).map(_normalize_linkedin)

    work["employee_count"] = _series_from_columns(
        work,
        ["employee_count", "effectif", "trancheeffectifsetablissement", "effectif_min"],
    ).map(_normalize_employee_bucket)

    work["revenue_range"] = _series_from_columns(
        work,
        ["revenue_range", "ca_brut", "chiffre_affaires", "chiffre_affaires_range"],
    ).replace("", pd.NA)

    work["date_creation"] = _series_from_columns(
        work,
        ["date_creation", "datecreationetablissement", "date_de_creation"],
    ).replace("", pd.NA)

    work["naf_code"] = _series_from_columns(
        work,
        ["naf_code", "naf", "code_naf", "ape", "code_apet"],
    ).replace("", pd.NA)
    work["naf"] = _series_from_columns(
        work,
        ["naf", "libelle_naf", "libellenomenclatureactiviteprincipaleetablissement"],
    ).replace("", pd.NA)

    email_series = _series_from_columns(
        work,
        ["email", "best_email", "contact_email", "emails"],
    ).map(_extract_first_token)
    work["email"] = email_series.replace("", pd.NA)

    phone_series = _series_from_columns(
        work,
        ["telephone", "phone", "phones", "contact_phone"],
    ).map(_extract_first_token)
    work["telephone"] = phone_series.replace("", pd.NA)

    work["etat_administratif"] = _series_from_columns(
        work,
        ["etat_administratif", "etatadministratifetablissement"],
    ).replace("", pd.NA)

    work["score_quality"] = work.get("score_quality")

    work["tech_stack"] = _series_from_columns(
        work,
        ["tech_stack", "technologies", "stacks"],
    ).map(_normalize_tech_stack)

    selected_columns = [col for col in PREMIUM_COLUMNS if col in work.columns]
    filtered = work[selected_columns].copy()
    for column in filtered.columns:
        if pd.api.types.is_object_dtype(filtered[column]):
            filtered[column] = filtered[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )

    key_subset: Optional[list[str]] = None
    if "siret" in filtered.columns and filtered["siret"].notna().any():
        key_subset = ["siret"]
    elif "domain" in filtered.columns and filtered["domain"].notna().any():
        key_subset = ["domain"]

    if key_subset:
        initial_count = len(filtered)
        filtered = filtered.drop_duplicates(subset=key_subset, keep="first")
        if len(filtered) < initial_count:
            log.info(
                "Deduplicated premium dataset on %s: %d -> %d rows",
                key_subset[0],
                initial_count,
                len(filtered),
            )

    return filtered.reset_index(drop=True)


def _compute_coverage(df: pd.DataFrame) -> dict[str, float]:
    total = len(df) or 1
    def _presence(column: str) -> float:
        if column not in df.columns:
            return 0.0
        series = df[column].fillna("").astype(str).str.strip()
        mask = series != ""
        return round(float(mask.sum() * 100.0 / total), 2)

    key_mask = pd.Series(False, index=df.index)
    if "siret" in df.columns:
        key_mask |= df["siret"].fillna("").astype(str).str.strip() != ""
    if "domain" in df.columns:
        key_mask |= df["domain"].fillna("").astype(str).str.strip() != ""

    contact_mask = pd.Series(False, index=df.index)
    if "email" in df.columns:
        contact_mask |= df["email"].fillna("").astype(str).str.strip() != ""
    if "telephone" in df.columns:
        contact_mask |= df["telephone"].fillna("").astype(str).str.strip() != ""

    return {
        "key_coverage_pct": round(float(key_mask.sum() * 100.0 / total), 2),
        "company_name_pct": _presence("company_name"),
        "adresse_complete_pct": _presence("adresse_complete"),
        "contact_pct": round(float(contact_mask.sum() * 100.0 / total), 2),
    }


def run_finalize_premium_dataset(cfg: dict, ctx: dict) -> dict:
    logger = ctx.get("logger") or LOGGER
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))

    parquet_path = outdir / "dataset_enriched.parquet"
    csv_path = outdir / "dataset_enriched.csv"

    if parquet_path.exists():
        dataset = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        dataset = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("Unable to locate dataset_enriched.[parquet|csv] for premium filtering")

    premium_df = filter_premium_columns(dataset, logger=logger)
    present_columns = [col for col in PREMIUM_COLUMNS if col in premium_df.columns]
    non_empty_columns = [
        col
        for col in present_columns
        if not premium_df[col].isna().all()
        and not (
            premium_df[col].fillna("").astype(str).str.strip() == ""
        ).all()
    ]
    if len(non_empty_columns) < 8:
        logger.warning(
            "Premium dataset exposes only %d informative columns; keeping full export intact",
            len(non_empty_columns),
            len(PREMIUM_COLUMNS),
        )
        return {
            "status": "WARN",
            "reason": "insufficient_columns",
            "columns": present_columns,
            "informative_columns": non_empty_columns,
            "rows": len(dataset),
        }

    premium_csv = outdir / "dataset.csv"
    premium_parquet = outdir / "dataset.parquet"

    premium_df.to_csv(premium_csv, index=False, encoding="utf-8")
    premium_df.to_parquet(premium_parquet, index=False)

    coverage = _compute_coverage(premium_df)

    logger.info(
        "Premium dataset ready | rows=%d | columns=%d (%d informative) | key_coverage=%.2f%% | company=%.2f%% | adresse=%.2f%% | contact=%.2f%%",
        len(premium_df),
        len(present_columns),
        len(non_empty_columns),
        coverage["key_coverage_pct"],
        coverage["company_name_pct"],
        coverage["adresse_complete_pct"],
        coverage["contact_pct"],
    )

    return {
        "status": "OK",
        "rows": len(premium_df),
        "columns": present_columns,
        "informative_columns": non_empty_columns,
        "csv": str(premium_csv),
        "parquet": str(premium_parquet),
        "coverage": coverage,
    }


__all__ = [
    "PREMIUM_COLUMNS",
    "filter_premium_columns",
    "run_finalize_premium_dataset",
]
