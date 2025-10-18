#!/usr/bin/env python3
"""
Utility script to train the domain prediction model from enriched datasets.

Usage examples:

    python train_domain_predictor.py --outdir out --model-path models/domain_predictor.joblib
    python train_domain_predictor.py --input out/enriched_domain.parquet --limit 50000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.domain_predictor import DEFAULT_MODEL_PATH, train_domain_predictor

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _resolve_input_path(outdir: Path, override: Optional[str]) -> Path:
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"Input dataset not found: {path}")
        return path
    for candidate in (
        outdir / "google_maps_enriched.parquet",
        outdir / "enriched_domain.parquet",
        outdir / "domains_enriched.parquet",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No suitable dataset found in {outdir}. Expected google_maps_enriched.parquet or enriched_domain.parquet."
    )


def _load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _ensure_domain_column(df: pd.DataFrame) -> pd.DataFrame:
    if "domain_true" in df.columns:
        return df
    for column in ("domain_root", "domain", "siteweb", "site_web"):
        if column in df.columns:
            df = df.copy()
            df["domain_true"] = df[column]
            return df
    raise ValueError("Input dataset must contain one of: domain_true, domain_root, domain, siteweb, site_web")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the domain prediction model.")
    parser.add_argument("--outdir", type=Path, default=Path("out"), help="Directory containing enriched datasets.")
    parser.add_argument("--input", type=str, default=None, help="Explicit path to the training dataset.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Destination path for the trained model.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick experimentation.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger("train_domain_predictor")

    dataset_path = _resolve_input_path(args.outdir, args.input)
    logger.info("Loading dataset from %s", dataset_path)
    df = _load_dataset(dataset_path)
    df = _ensure_domain_column(df)

    if args.limit:
        df = df.head(args.limit).copy()
        logger.info("Dataset truncated to %d rows for training preview", len(df))

    predictor = train_domain_predictor(df, model_path=args.model_path, logger=logger)

    logger.info(
        "Model saved to %s | samples=%d | positives=%d | accuracy=%.3f | roc_auc=%.3f | avg_precision=%.3f",
        args.model_path,
        predictor.metadata.get("training_samples", 0),
        predictor.metadata.get("positive_samples", 0),
        predictor.metadata.get("accuracy", float("nan")),
        predictor.metadata.get("roc_auc", float("nan")),
        predictor.metadata.get("average_precision", float("nan")),
    )


if __name__ == "__main__":
    main()
