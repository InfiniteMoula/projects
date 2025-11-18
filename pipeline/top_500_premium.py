"""Pipeline helpers to generate the top_500_premium dataset."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from utils import io, pipeline

LOGGER = logging.getLogger("pipeline.top_500_premium")
STATE_KEY = "_top_500_premium_state"

DEFAULT_CONFIG = {
    "input_parquet": "data/enriched/all_companies_enriched.parquet",
    "filter_expr": "is_sellable == True",
    "score_column": "score_global",
    "score_default": 0.0,
    "limit": 500,
    "output_csv": "exports/top_500_premium.csv",
    "output_encoding": "utf-8-sig",
    "columns_order": [
        "company_name",
        "enseigne",
        "siren",
        "siret",
        "naf_code",
        "naf_label",
        "code_naf",
        "activite",
        "date_creation",
        "employee_count",
        "revenue_range",
        "adresse_complete",
        "code_postal",
        "ville",
        "departement",
        "region",
        "pays",
        "is_sellable",
        "score_global",
        "score_quality",
        "score_email",
        "score_phone",
        "site_web",
        "domain",
        "email",
        "telephone",
        "linkedin_company_url",
        "linkedin_url",
    ],
}


def _get_logger(ctx: dict | None) -> logging.Logger:
    return (ctx or {}).get("logger") or LOGGER


def _get_state(ctx: dict) -> dict:
    return ctx.setdefault(STATE_KEY, {})


def _require_frame(state: dict) -> pd.DataFrame:
    df = state.get("frame")
    if df is None:
        raise RuntimeError("Dataset not loaded yet; run top500.load_dataset first")
    return df


def _get_dataset_config(job: dict) -> dict:
    config = job.get("top_500_premium") or job.get("dataset_pipeline") or {}
    if not isinstance(config, dict):
        raise ValueError("Invalid top_500_premium configuration block")
    merged = {**DEFAULT_CONFIG, **config}
    return merged


def _resolve_input_path(config: dict, ctx: dict) -> Path:
    ctx_input = ctx.get("input_path")
    if ctx_input:
        return Path(ctx_input)
    return Path(config["input_parquet"]).expanduser()


def filter_rows(df: pd.DataFrame, condition_expr: str) -> pd.DataFrame:
    """Return a filtered copy of *df* using a pandas expression."""
    if not condition_expr:
        return df.copy()
    try:
        mask = df.eval(condition_expr, engine="python")
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid filter expression '{condition_expr}': {exc}") from exc
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return df[mask].copy()


def sort_values(df: pd.DataFrame, by: Sequence[str] | str, ascending: bool | Sequence[bool]) -> pd.DataFrame:
    """Return a sorted copy of *df*."""
    columns: Sequence[str]
    if isinstance(by, str):
        columns = [by]
    else:
        columns = list(by)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Cannot sort on missing columns: {missing}")
    return df.sort_values(by=columns, ascending=ascending).reset_index(drop=True)


def head(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return the first *n* rows of *df*."""
    if n <= 0:
        return df.head(0).copy()
    return df.head(n).copy().reset_index(drop=True)


def reorder_columns(df: pd.DataFrame, columns: Iterable[str] | None) -> pd.DataFrame:
    """Reorder dataframe columns while keeping unknown columns at the end."""
    if not columns:
        return df.copy()
    ordered: list[str] = []
    for column in columns:
        if column in df.columns and column not in ordered:
            ordered.append(column)
    tail = [column for column in df.columns if column not in ordered]
    return df.loc[:, ordered + tail].copy()


def load_enriched_dataset(job: dict, ctx: dict) -> dict:
    config = _get_dataset_config(job)
    logger = _get_logger(ctx)
    input_path = _resolve_input_path(config, ctx)
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")
    logger.info("Loading enriched dataset from %s", input_path)
    df = pd.read_parquet(input_path)
    sample = int(ctx.get("sample") or 0)
    if sample > 0:
        df = df.head(sample)
        logger.info("Sample mode enabled -> using first %d rows", len(df))
    state = _get_state(ctx)
    state["frame"] = df
    state["source_path"] = str(input_path)
    return {"status": "OK", "rows": len(df), "columns": list(df.columns)}


def filter_sellable(job: dict, ctx: dict) -> dict:
    config = _get_dataset_config(job)
    logger = _get_logger(ctx)
    state = _get_state(ctx)
    df = _require_frame(state)
    filtered = filter_rows(df, config["filter_expr"])
    logger.info("Filtered dataset with %s -> %d rows", config["filter_expr"], len(filtered))
    state["frame"] = filtered
    return {"status": "OK", "rows": len(filtered)}


def ensure_score(job: dict, ctx: dict) -> dict:
    config = _get_dataset_config(job)
    logger = _get_logger(ctx)
    state = _get_state(ctx)
    df = _require_frame(state).copy()
    column = config["score_column"]
    default_value = float(config["score_default"])
    created = False
    if column not in df.columns:
        df[column] = default_value
        created = True
    else:
        df[column] = df[column].fillna(default_value)
    state["frame"] = df
    if created:
        logger.warning("Column %s missing -> filled with default %s", column, default_value)
    else:
        logger.info("Ensured %s column completeness", column)
    return {"status": "OK", "column": column, "default": default_value}


def sort_by_score(job: dict, ctx: dict) -> dict:
    config = _get_dataset_config(job)
    logger = _get_logger(ctx)
    state = _get_state(ctx)
    df = _require_frame(state)
    sorted_df = sort_values(df, config["score_column"], ascending=False)
    state["frame"] = sorted_df
    logger.info("Sorted dataset by %s (desc)", config["score_column"])
    return {"status": "OK", "rows": len(sorted_df)}


def select_head(job: dict, ctx: dict) -> dict:
    config = _get_dataset_config(job)
    logger = _get_logger(ctx)
    state = _get_state(ctx)
    df = _require_frame(state)
    top_n = int(config["limit"])
    limited = head(df, top_n)
    state["frame"] = limited
    logger.info("Selected top %d rows", len(limited))
    return {"status": "OK", "rows": len(limited)}


def reorder_for_export(job: dict, ctx: dict) -> dict:
    config = _get_dataset_config(job)
    logger = _get_logger(ctx)
    state = _get_state(ctx)
    df = _require_frame(state)
    reordered = reorder_columns(df, config.get("columns_order"))
    state["frame"] = reordered
    logger.info("Reordered columns for premium export")
    return {"status": "OK", "columns": list(reordered.columns)}


def export_csv(job: dict, ctx: dict) -> dict:
    config = _get_dataset_config(job)
    logger = _get_logger(ctx)
    state = _get_state(ctx)
    df = _require_frame(state)
    output_path = Path(config["output_csv"]).expanduser()
    encoding = config["output_encoding"]
    io.ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False, encoding=encoding)
    logger.info(
        "top_500_premium export ready -> %s (%d rows, %d columns)",
        output_path,
        len(df),
        len(df.columns),
    )
    ctx.setdefault("exports", {})["top_500_premium_csv"] = str(output_path)
    return {
        "status": "OK",
        "rows": len(df),
        "columns": list(df.columns),
        "output_csv": str(output_path),
        "encoding": encoding,
    }


__all__ = [
    "filter_rows",
    "sort_values",
    "head",
    "reorder_columns",
    "load_enriched_dataset",
    "filter_sellable",
    "ensure_score",
    "sort_by_score",
    "select_head",
    "reorder_for_export",
    "export_csv",
]
