"""Clean and consolidate contact information extracted from crawls."""
from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from utils import io
from utils.url import registered_domain

EMAIL_DOMAIN_RE = re.compile(r"^[^@]+@([A-Za-z0-9.-]+)$")


def _normalize_siren(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 9:
        return digits[:9]
    return None


def _coerce_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, (set, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return []
        if trimmed.startswith("[") and trimmed.endswith("]"):
            try:
                parsed = ast.literal_eval(trimmed)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return [str(v).strip() for v in parsed if str(v).strip()]
        return [trimmed]
    return []


def _merge_sets(target: set, values: Iterable[str]) -> None:
    for item in values:
        if item:
            target.add(str(item).strip())


def _email_matches_domain(email: str, domain: str) -> bool:
    if not email or not domain:
        return False
    match = EMAIL_DOMAIN_RE.match(email)
    if not match:
        return False
    email_domain = match.group(1).lower()
    site_domain = domain.lower()
    if email_domain == site_domain:
        return True
    if email_domain.endswith("." + site_domain.lstrip(".")):
        return True
    return registered_domain(email_domain) == registered_domain(site_domain)


def _select_best(values: Sequence[str]) -> Optional[str]:
    for value in values:
        if value:
            return value
    return None


def _collect_company_key(row: pd.Series) -> Tuple[str, Optional[str], Optional[str]]:
    siren = _normalize_siren(row.get("siren"))
    if not siren:
        sirens_extracted = _coerce_list(row.get("sirens_extracted"))
        for candidate in sirens_extracted:
            siren = _normalize_siren(candidate)
            if siren:
                break
    domain = str(row.get("domain") or "").strip().lower() or None
    key = siren or domain
    return key or "", siren, domain


def run(cfg: dict, ctx: dict) -> dict:
    logger = ctx.get("logger")
    outdir = Path(ctx["outdir"])
    contacts_path = outdir / "contacts" / "contacts.parquet"
    if not contacts_path.exists():
        if logger:
            logger.info("clean_contacts: contacts.parquet not found at %s", contacts_path)
        return {"status": "SKIPPED", "reason": "NO_CONTACTS"}

    df = pd.read_parquet(contacts_path)
    if df.empty:
        no_contact_path = io.ensure_dir(outdir / "contacts") / "no_contact.csv"
        pd.DataFrame(columns=["siren", "domain", "denomination", "top_url"]).to_csv(no_contact_path, index=False)
        if logger:
            logger.info("clean_contacts: no contacts to clean (empty input)")
        return {"status": "SKIPPED", "reason": "EMPTY_CONTACTS"}

    aggregated: Dict[str, dict] = {}
    for _, row in df.iterrows():
        key, siren, domain = _collect_company_key(row)
        if not key:
            continue
        entry = aggregated.setdefault(
            key,
            {
                "siren": siren,
                "domain": domain,
                "denomination": None,
                "top_url": None,
                "best_page": None,
                "best_status": None,
                "confidence": None,
                "query": None,
                "rank": None,
                "discovered_at": None,
                "emails": set(),
                "phones": set(),
                "social_links": {},
                "rcs": set(),
                "legal_managers": set(),
                "addresses": set(),
                "sirets": set(),
                "sirens_extracted": set(),
                "best_email": None,
                "best_email_type": None,
            },
        )

        if siren and not entry["siren"]:
            entry["siren"] = siren
        if domain and not entry["domain"]:
            entry["domain"] = domain

        denomination = str(row.get("denomination") or "").strip()
        if denomination and not entry["denomination"]:
            entry["denomination"] = denomination

        top_url = str(row.get("top_url") or "").strip()
        if top_url and not entry["top_url"]:
            entry["top_url"] = top_url

        best_page = str(row.get("best_page") or "").strip()
        if best_page and not entry["best_page"]:
            entry["best_page"] = best_page

        best_status = row.get("best_status")
        if entry["best_status"] is None and pd.notna(best_status):
            entry["best_status"] = int(best_status)

        confidence = row.get("confidence")
        if entry["confidence"] is None and pd.notna(confidence):
            entry["confidence"] = float(confidence)

        query = str(row.get("query") or "").strip()
        if query and not entry["query"]:
            entry["query"] = query

        rank = row.get("rank")
        if entry["rank"] is None and pd.notna(rank):
            entry["rank"] = int(rank)

        discovered = str(row.get("discovered_at") or "").strip()
        if discovered and not entry["discovered_at"]:
            entry["discovered_at"] = discovered

        emails = [email.lower() for email in _coerce_list(row.get("emails")) if email]
        domain_for_check = entry["domain"] or domain or ""
        for email in emails:
            if not domain_for_check or _email_matches_domain(email, domain_for_check):
                entry["emails"].add(email)

        phones = _coerce_list(row.get("phones"))
        _merge_sets(entry["phones"], phones)

        social_links = row.get("social_links") or {}
        if isinstance(social_links, str):
            try:
                parsed = ast.literal_eval(social_links)
            except (SyntaxError, ValueError):
                parsed = {}
            if isinstance(parsed, dict):
                social_links = parsed
            else:
                social_links = {}
        if isinstance(social_links, dict):
            for network, links in social_links.items():
                bucket = entry["social_links"].setdefault(network, set())
                _merge_sets(bucket, _coerce_list(links))

        _merge_sets(entry["rcs"], _coerce_list(row.get("rcs")))
        _merge_sets(entry["legal_managers"], _coerce_list(row.get("legal_managers")))
        _merge_sets(entry["addresses"], _coerce_list(row.get("addresses")))
        _merge_sets(entry["sirets"], _coerce_list(row.get("sirets")))
        _merge_sets(entry["sirens_extracted"], _coerce_list(row.get("sirens_extracted")))

        row_best_email = str(row.get("best_email") or "").strip().lower()
        row_email_type = str(row.get("email_type") or "").strip().lower() or None
        if row_best_email and (not domain_for_check or _email_matches_domain(row_best_email, domain_for_check)):
            current_best = entry["best_email"]
            current_type = entry["best_email_type"] or ""
            candidate_score = 0 if row_email_type == "nominative" else 1
            current_score = 0 if current_type == "nominative" else 1
            if current_best is None or candidate_score < current_score:
                entry["best_email"] = row_best_email
                entry["best_email_type"] = row_email_type

    cleaned_rows: List[dict] = []
    no_contacts: List[dict] = []
    for entry in aggregated.values():
        emails_sorted = sorted(entry["emails"])
        phones_sorted = sorted(entry["phones"])
        social_links_sorted = {
            network: sorted(values) for network, values in entry["social_links"].items() if values
        }
        record = {
            "siren": entry["siren"],
            "domain": entry["domain"],
            "denomination": entry["denomination"],
            "top_url": entry["top_url"],
            "best_page": entry["best_page"],
            "best_status": entry["best_status"],
            "best_email": entry["best_email"] if entry["best_email"] in emails_sorted else _select_best(emails_sorted),
            "email_type": entry["best_email_type"],
            "emails": emails_sorted,
            "phones": phones_sorted,
            "social_links": social_links_sorted,
            "rcs": sorted(entry["rcs"]),
            "legal_managers": sorted(entry["legal_managers"]),
            "addresses": sorted(entry["addresses"]),
            "sirets": sorted(entry["sirets"]),
            "sirens_extracted": sorted(entry["sirens_extracted"]),
            "confidence": entry["confidence"],
            "query": entry["query"],
            "rank": entry["rank"],
            "discovered_at": entry["discovered_at"],
        }
        has_contacts = bool(record["emails"] or record["phones"])
        if not has_contacts:
            no_contacts.append(
                {
                    "siren": record["siren"],
                    "domain": record["domain"],
                    "denomination": record["denomination"],
                    "top_url": record["top_url"],
                }
            )
        cleaned_rows.append(record)

    cleaned_df = pd.DataFrame(cleaned_rows)
    contacts_dir = io.ensure_dir(outdir / "contacts")
    clean_parquet = contacts_dir / "contacts_clean.parquet"
    clean_csv = contacts_dir / "contacts_clean.csv"
    cleaned_df.to_parquet(clean_parquet, index=False)
    cleaned_df.to_csv(clean_csv, index=False)

    no_contact_df = pd.DataFrame(no_contacts)
    no_contact_path = contacts_dir / "no_contact.csv"
    no_contact_df.to_csv(no_contact_path, index=False)

    total = len(cleaned_df)
    enriched = int((cleaned_df["emails"].apply(bool) | cleaned_df["phones"].apply(bool)).sum())
    summary = f"clean_contacts: {enriched}/{total} companies enriched"
    if logger:
        logger.info(summary)
    else:
        print(summary)

    return {
        "status": "OK",
        "output": str(clean_parquet),
        "records": total,
        "enriched": enriched,
        "no_contact": len(no_contact_df),
    }


if __name__ == "__main__":
    raise SystemExit("clean_contacts is intended to be invoked via builder run pipeline")

