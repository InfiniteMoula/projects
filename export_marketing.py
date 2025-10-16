"""CLI to generate marketing-friendly exports from the enriched dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from marketing_exports import generate_marketing_exports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate top contactable companies export for marketing teams",
    )
    parser.add_argument(
        "--outdir",
        default="out",
        type=Path,
        help="Pipeline output directory containing dataset_enriched.[parquet|csv]",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Override path to dataset_enriched file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of contactable companies to export",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = generate_marketing_exports(
        outdir=args.outdir,
        dataset_path=args.dataset,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
