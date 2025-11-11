"""Correlation module computing a coherence score across contact fields."""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence

import pandas as pd

from . import rules

LOGGER = logging.getLogger("correlation.correlation")

DEFAULT_BASE_SCORE = 50.0
DEFAULT_REPORT_PATH = Path("reports/report_correlation.json")
COUNTRY_FIELD_CANDIDATES = ("country_code", "code_pays", "code_pays_iso", "pays", "country")
PHONE_FIELD_CANDIDATES = ("telephone_norm", "telephone", "phone")


def _get_cfg_value(cfg: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        return getattr(cfg, key)
    return default


def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, value))


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "score_coherence" not in df.columns:
        df["score_coherence"] = pd.Series(dtype="float64")
    if "coherence_flags" not in df.columns:
        df["coherence_flags"] = pd.Series(dtype="object")
    return df


def _score_distribution(scores: Sequence[float]) -> MutableMapping[str, int]:
    buckets = {
        "0-19": 0,
        "20-39": 0,
        "40-59": 0,
        "60-79": 0,
        "80-100": 0,
    }
    for score in scores:
        if score < 20:
            buckets["0-19"] += 1
        elif score < 40:
            buckets["20-39"] += 1
        elif score < 60:
            buckets["40-59"] += 1
        elif score < 80:
            buckets["60-79"] += 1
        else:
            buckets["80-100"] += 1
    return buckets


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _select_first_present(row: Any, candidates: Sequence[str]) -> Optional[object]:
    for field in candidates:
        if hasattr(row, field):
            value = getattr(row, field)
            if value not in (None, ""):
                return value
    return None


def _resolve_outdir(ctx: Mapping[str, Any]) -> Path:
    outdir_path = ctx.get("outdir_path")
    if outdir_path:
        return Path(outdir_path)
    outdir = ctx.get("outdir")
    if outdir:
        return Path(outdir)
    return Path.cwd()


def _load_contacts_dataframe(ctx: Mapping[str, Any], cfg: Mapping[str, Any], *, logger: logging.Logger) -> tuple[pd.DataFrame, Path]:
    outdir = _resolve_outdir(ctx)
    source_override = cfg.get("source") if isinstance(cfg, Mapping) else None
    if source_override:
        source_path = Path(source_override)
        if not source_path.is_absolute():
            source_path = outdir / source_path
    else:
        source_path = outdir / "contacts_enriched.parquet"
        if not source_path.exists():
            source_path = outdir / "contacts_enriched.csv"

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if source_path.suffix.lower() == ".csv":
        df = pd.read_csv(source_path)
    else:
        df = pd.read_parquet(source_path)

    logger.debug("Loaded %d rows for correlation from %s", len(df), source_path)
    return df, source_path


def _prepare_step_cfg(job_cfg: Mapping[str, Any], ctx: Mapping[str, Any]) -> Mapping[str, Any]:
    step_cfg = {}
    if isinstance(job_cfg, Mapping):
        correlation_cfg = job_cfg.get("correlation") or {}
        if isinstance(correlation_cfg, Mapping):
            step_cfg.update(correlation_cfg)
    report_dir = ctx.get("directories", {}).get("reports")
    if report_dir is None:
        report_dir = _resolve_outdir(ctx) / "reports"
    report_path = Path(step_cfg.get("report_path") or report_dir / "correlation_summary.json")
    step_cfg["report_path"] = str(report_path)
    return step_cfg


def _run_dataframe(df_in: pd.DataFrame | None, cfg: Mapping[str, Any] | None) -> pd.DataFrame:
    cfg = cfg or {}
    df_in = df_in if df_in is not None else pd.DataFrame()
    df_out = _ensure_columns(df_in.copy())

    if df_out.empty:
        report_path = Path(_get_cfg_value(cfg, "report_path", DEFAULT_REPORT_PATH))
        payload = {
            "total_rows": 0,
            "average_score": 0.0,
            "incoherent_rows": 0,
            "score_distribution": _score_distribution(()),
            "flag_counts": {},
            "top_incoherences": [],
        }
        _write_report(report_path, payload)
        LOGGER.info("correlation summary | rows=0 | incoherences=0 | avg_score=0.00")
        return df_out

    base_score = float(_get_cfg_value(cfg, "base_score", DEFAULT_BASE_SCORE))
    report_path = Path(_get_cfg_value(cfg, "report_path", DEFAULT_REPORT_PATH))
    default_country = _get_cfg_value(cfg, "default_country", "FR")
    overrides = _get_cfg_value(cfg, "country_overrides", None)
    overrides = dict(overrides) if isinstance(overrides, Mapping) else None
    country_field = _get_cfg_value(cfg, "country_field", None)
    if not country_field:
        for candidate in COUNTRY_FIELD_CANDIDATES:
            if candidate in df_out.columns:
                country_field = candidate
                break
    top_n = int(_get_cfg_value(cfg, "top_incoherences", 5) or 5)

    scores: List[float] = []
    negative_flags_counter: Counter[str] = Counter()
    row_summaries: List[dict[str, Any]] = []

    for row in df_out.itertuples(index=True):
        idx = row.Index
        site_value = getattr(row, "site_web", None)
        email_value = getattr(row, "email", None)
        linkedin_value = getattr(row, "linkedin_url", None)
        phone_value = _select_first_present(row, PHONE_FIELD_CANDIDATES)
        country_value = getattr(row, country_field, None) if country_field else None
        expected_country = rules.normalize_country(country_value, default_country, overrides)

        rule_results: List[rules.RuleResult] = []
        for outcome in (
            rules.email_domain_alignment(email_value, site_value),
            rules.generic_email_penalty(email_value),
            rules.phone_country_alignment(phone_value, expected_country),
            rules.linkedin_site_alignment(linkedin_value, site_value),
        ):
            if outcome is not None:
                rule_results.append(outcome)

        score = base_score + sum(result.delta for result in rule_results)
        score = _clamp(score)
        flags = [result.flag for result in rule_results if result.flag]
        negative_flags = [result.flag for result in rule_results if result.flag and result.delta < 0]

        df_out.at[idx, "score_coherence"] = score
        df_out.at[idx, "coherence_flags"] = flags

        scores.append(score)
        negative_flags_counter.update(flag for flag in negative_flags if flag)
        row_summaries.append(
            {
                "index": idx,
                "score": score,
                "flags": flags,
                "negative_flags": negative_flags,
                "siren": getattr(row, "siren", None),
                "siret": getattr(row, "siret", None),
            }
        )

    total_rows = len(df_out)
    incoherent_rows = sum(1 for summary in row_summaries if summary["negative_flags"])
    average_score = sum(scores) / total_rows if total_rows else 0.0
    distribution = _score_distribution(scores)
    flag_counts = {flag: count for flag, count in negative_flags_counter.most_common()}

    incoherent_entries = [summary for summary in row_summaries if summary["negative_flags"]]
    incoherent_entries.sort(key=lambda item: item["score"])
    top_incoherences = [
        {
            "siren": entry["siren"],
            "siret": entry["siret"],
            "score": entry["score"],
            "flags": entry["negative_flags"],
        }
        for entry in incoherent_entries[: max(top_n, 0)]
    ]

    report_payload = {
        "total_rows": total_rows,
        "average_score": round(average_score, 2),
        "incoherent_rows": incoherent_rows,
        "score_distribution": distribution,
        "flag_counts": flag_counts,
        "top_incoherences": top_incoherences,
    }
    _write_report(report_path, report_payload)

    LOGGER.info(
        "correlation summary | rows=%s | incoherences=%s | avg_score=%.2f",
        total_rows,
        incoherent_rows,
        average_score,
    )
    return df_out


def _run_pipeline(job_cfg: Mapping[str, Any] | None, ctx: Mapping[str, Any] | None) -> Mapping[str, Any]:
    ctx = ctx or {}
    logger = ctx.get("logger", LOGGER)
    job_cfg = job_cfg or {}

    try:
        df_in, source_path = _load_contacts_dataframe(ctx, job_cfg, logger=logger)
    except FileNotFoundError as exc:
        logger.warning("correlation.checks skipped: missing input %s", exc)
        return {"status": "SKIPPED", "reason": "MISSING_INPUT", "source": str(exc)}
    except Exception as exc:  # pragma: no cover - IO defensive
        logger.exception("Failed to load contacts data for correlation")
        return {"status": "FAIL", "error": str(exc)}

    step_cfg = _prepare_step_cfg(job_cfg, ctx)
    df_out = _run_dataframe(df_in, step_cfg)

    outdir = _resolve_outdir(ctx)
    output_path = outdir / "contacts_correlation.parquet"
    df_out.to_parquet(output_path, index=False)

    status = "OK"
    reason = None
    if df_out.empty:
        status = "SKIPPED"
        reason = "EMPTY_INPUT"
        logger.info("correlation.checks skipped: empty dataset (%s)", source_path.name)
    else:
        logger.info(
            "correlation.checks completed: rows=%d | report=%s",
            len(df_out),
            step_cfg.get("report_path"),
        )

    return {
        "status": status,
        "rows": len(df_out),
        "file": str(output_path),
        "source": str(source_path),
        "report": str(step_cfg.get("report_path")),
        **({"reason": reason} if reason else {}),
    }


def run(first_arg: Any, second_arg: Mapping[str, Any] | None = None):
    """Entry point compatible with both tests (DataFrame input) and CLI (cfg/context)."""

    if isinstance(first_arg, pd.DataFrame) or first_arg is None:
        return _run_dataframe(first_arg, second_arg)
    return _run_pipeline(first_arg, second_arg)


__all__ = ["run"]
