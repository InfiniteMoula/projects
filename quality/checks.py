
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from .validation import (
    FieldValidation,
    validate_email,
    validate_linkedin_url,
    validate_site_web,
    validate_telephone,
)
from utils import io
from utils.email_validation import SMTPVerificationResult, SMTPVerificationStatus
from utils.parquet import iter_batches


class FieldStats:
    """Accumulate validation metrics for a single field."""

    def __init__(self, sample_size: int = 5) -> None:
        self.present = 0
        self.valid = 0
        self.invalid = 0
        self.missing = 0
        self.sample_size = sample_size
        self.invalid_examples: list[dict] = []
        self.reason_counts: Counter[str] = Counter()
        self.smtp_status_counts: Counter[str] = Counter()

    def add_missing(self) -> None:
        self.missing += 1

    def add_result(self, raw_value: str, result: FieldValidation) -> None:
        self.present += 1
        if result.details:
            status = result.details.get("smtp_status")
            if isinstance(status, str):
                self.smtp_status_counts[status] += 1

        if result.is_valid:
            self.valid += 1
            return
        self.invalid += 1
        reason = result.reason or "unknown"
        self.reason_counts[reason] += 1
        if len(self.invalid_examples) < self.sample_size:
            self.invalid_examples.append(
                {"value": raw_value[:120], "reason": reason, "details": result.details or {}}
            )

    def to_payload(self) -> Dict[str, object]:
        total = self.present + self.missing
        return {
            "total": total,
            "present": self.present,
            "missing": self.missing,
            "valid": self.valid,
            "invalid": self.invalid,
            "invalid_rate": (self.invalid / self.present) if self.present else 0.0,
            "reason_counts": dict(self.reason_counts),
            "invalid_examples": self.invalid_examples,
            "smtp_status_counts": dict(self.smtp_status_counts),
        }


def _is_blank(value) -> bool:
    if isinstance(value, str):
        return not value.strip()
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return value is None


def run(cfg, ctx):
    outdir = Path(ctx.get("outdir_path") or ctx.get("outdir"))
    candidates = [outdir / "enriched_email.parquet", outdir / "normalized.parquet"]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        return {"status": "WARN", "error": "no data for quality checks"}

    issues = []
    siren_nulls = 0
    cp_bad = 0
    smtp_cfg = {}
    if isinstance(cfg, dict):
        smtp_cfg = cfg.get("smtp_verification") or {}
    smtp_enabled = bool(smtp_cfg.get("enabled", True))
    try:
        smtp_timeout = float(smtp_cfg.get("timeout", 10.0))
    except (TypeError, ValueError):
        smtp_timeout = 10.0
    validation_stats = {
        "site_web": FieldStats(),
        "email": FieldStats(),
        "telephone": FieldStats(),
        "linkedin_url": FieldStats(),
    }
    email_mx_inconclusive = 0
    email_mx_cache: Dict[str, Optional[bool]] = {}
    email_smtp_cache: Dict[str, SMTPVerificationResult] = {} if smtp_enabled else {}
    email_smtp_douteux = 0
    email_smtp_invalid = 0
    email_smtp_catch_all = 0

    columns: Iterable[str] = [
        "siren",
        "cp",
        "site_web",
        "email",
        "telephone",
        "linkedin_url",
    ]

    for df in iter_batches(source, columns=columns):
        if "siren" in df.columns:
            siren_nulls += int(df["siren"].isna().sum())
        if "cp" in df.columns:
            cp_series = df["cp"].astype("string").fillna("")
            cp_bad += int((~cp_series.str.match(r"^\d{2,5}$", na=False)).sum())

        if "site_web" in df.columns:
            for value in df["site_web"]:
                if _is_blank(value):
                    validation_stats["site_web"].add_missing()
                    continue
                raw = str(value).strip()
                result = validate_site_web(raw)
                validation_stats["site_web"].add_result(raw, result)

        if "email" in df.columns:
            for value in df["email"]:
                if _is_blank(value):
                    validation_stats["email"].add_missing()
                    continue
                raw = str(value).strip()
                result = validate_email(
                    raw,
                    check_mx=True,
                    mx_cache=email_mx_cache,
                    check_smtp=smtp_enabled,
                    smtp_cache=email_smtp_cache if smtp_enabled else None,
                    smtp_timeout=smtp_timeout,
                )
                if result.details and result.details.get("mx_valid") is None:
                    email_mx_inconclusive += 1
                if smtp_enabled and result.details:
                    status = result.details.get("smtp_status")
                    reason = result.details.get("smtp_reason")
                    if status == SMTPVerificationStatus.DOUTEUX.value:
                        email_smtp_douteux += 1
                        if reason == "catch_all":
                            email_smtp_catch_all += 1
                    elif status == SMTPVerificationStatus.INVALIDE.value:
                        email_smtp_invalid += 1
                validation_stats["email"].add_result(raw, result)

        if "telephone" in df.columns:
            for value in df["telephone"]:
                if _is_blank(value):
                    validation_stats["telephone"].add_missing()
                    continue
                raw = str(value).strip()
                result = validate_telephone(raw)
                validation_stats["telephone"].add_result(raw, result)

        if "linkedin_url" in df.columns:
            for value in df["linkedin_url"]:
                if _is_blank(value):
                    validation_stats["linkedin_url"].add_missing()
                    continue
                raw = str(value).strip()
                result = validate_linkedin_url(raw)
                validation_stats["linkedin_url"].add_result(raw, result)

    if siren_nulls:
        issues.append({"key": "siren_nulls", "count": siren_nulls})
    if cp_bad:
        issues.append({"key": "cp_format", "count": cp_bad})

    validation_payload = {}
    for field, stats in validation_stats.items():
        summary = stats.to_payload()
        validation_payload[field] = summary
        if summary["invalid"]:
            issues.append({"key": f"{field}_invalid", "count": summary["invalid"]})
        if summary["missing"] and summary["present"]:
            issues.append({"key": f"{field}_missing", "count": summary["missing"]})

    if email_mx_inconclusive:
        issues.append({"key": "email_mx_lookup_inconclusive", "count": email_mx_inconclusive})
    if smtp_enabled:
        if email_smtp_invalid:
            issues.append({"key": "email_smtp_invalid", "count": email_smtp_invalid})
        if email_smtp_douteux:
            issues.append({"key": "email_smtp_douteux", "count": email_smtp_douteux})
        if email_smtp_catch_all:
            issues.append({"key": "email_smtp_catch_all", "count": email_smtp_catch_all})

    payload = {"issues": issues, "validation": validation_payload}
    out_path = outdir / "quality_checks.json"
    io.write_json(out_path, payload)
    return {"status": "OK", "file": str(out_path), "issues": issues}
