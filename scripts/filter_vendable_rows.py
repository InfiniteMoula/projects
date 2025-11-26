import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


def configure_logging(verbosity: int = 0) -> None:
    level = logging.INFO if verbosity == 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def load_dataframe(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    LOGGER.info("Loading input file: %s", input_path)

    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    LOGGER.info("Loaded %d rows and %d columns", len(df), len(df.columns))
    return df


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    LOGGER.info("Saving output file: %s", output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    LOGGER.info("Saved %d rows and %d columns", len(df), len(df.columns))


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names for more robust matching."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def get_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def filter_vendable_rows(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """
    Applique la définition "ligne vendable" (version 1).

    Règles :
    - siret, nom, code_naf non nuls
    - domain non nul / non vide
    - au moins un contact (email ou téléphone)
    """
    df = normalize_column_names(df)

    # Configuration par défaut des noms de colonnes possibles
    default_mapping: Dict[str, list[str]] = {
        "siret": ["siret"],
        "name": ["nom", "raison_sociale", "name"],
        "naf": ["code_naf", "naf", "ape"],
        "city": ["ville", "commune", "city"],
        "domain": ["domain", "domaine", "website", "site_web", "url"],
        "email": ["email", "mail", "courriel"],
        "phone": ["phone", "telephone", "tel", "phone_number"],
    }

    if column_mapping:
        for key, candidates in column_mapping.items():
            default_mapping[key] = candidates

    # Résolution des colonnes réelles
    resolved: Dict[str, Optional[str]] = {}
    for logical_name, candidates in default_mapping.items():
        col = get_first_existing_column(df, candidates)
        resolved[logical_name] = col
        LOGGER.debug("Resolved column '%s' -> %s", logical_name, col)

    required = ["siret", "name", "naf"]
    for key in required:
        if resolved[key] is None:
            raise KeyError(
                f"Required logical column '{key}' not found in dataframe. "
                f"Tried candidates: {default_mapping[key]}"
            )

    if resolved["domain"] is None:
        raise KeyError(
            "No domain column found. Tried candidates: "
            f"{default_mapping['domain']}"
        )

    # Conditions de base (identification)
    mask = (
        df[resolved["siret"]].notna()
        & df[resolved["name"]].notna()
        & df[resolved["naf"]].notna()
    )

    # Domain non nul / non vide
    domain_col = resolved["domain"]
    mask = mask & df[domain_col].notna() & (df[domain_col].astype(str).str.strip() != "")

    # Contact : email ou téléphone
    contact_mask = False
    if resolved["email"] is not None:
        contact_mask = contact_mask | df[resolved["email"]].notna()
    if resolved["phone"] is not None:
        contact_mask = contact_mask | df[resolved["phone"]].notna()

    if isinstance(contact_mask, bool):
        # Aucun des deux n'existe -> on garde uniquement sur domain + id
        LOGGER.warning(
            "No email/phone columns found. Vendable definition will ignore contact criteria."
        )
        final_mask = mask
    else:
        final_mask = mask & contact_mask

    before_count = len(df)
    filtered_df = df[final_mask].copy()
    after_count = len(filtered_df)

    LOGGER.info("Input rows: %d", before_count)
    LOGGER.info("Vendable rows: %d", after_count)
    if before_count > 0:
        LOGGER.info("Kept %.2f%% of rows", after_count * 100.0 / before_count)

    return filtered_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter vendable rows from an enriched companies dataset."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input CSV or Parquet file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output CSV or Parquet file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity level (can be used multiple times).",
    )

    args = parser.parse_args()
    configure_logging(args.verbose)

    input_path = Path(args.input)
    output_path = Path(args.output)

    df = load_dataframe(input_path)
    filtered_df = filter_vendable_rows(df)
    save_dataframe(filtered_df, output_path)


if __name__ == "__main__":
    main()
