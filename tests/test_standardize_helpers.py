import pandas as pd

from normalize import standardize


def test_extract_departement_handles_various_formats():
    series = pd.Series([
        "75001",   # standard 5-digit code
        "3400",    # 4-digit code missing leading zero
        "2A345",   # alphanumeric code for Corsica
        "",        # empty string should map to <NA>
        pd.NA,      # actual missing value
    ], dtype="string")

    result = standardize._extract_departement(series)

    assert list(result.astype(object)) == [
        "75",   # Paris
        "03",   # 03400 -> department 03
        "2A",  # Preserve alphanumeric department codes
        pd.NA,
        pd.NA,
    ]


def test_fr_tel_norm_normalizes_and_filters_placeholders():
    series = pd.Series([
        "01 23 45 67 89",          # classic formatting with spaces
        "123456789",               # 9 digits => assume missing leading zero
        "TELEPHONE NON RENSEIGNE",  # placeholder should be dropped
        "",
    ], dtype="string")

    normalized = standardize._fr_tel_norm(series)

    assert list(normalized.astype(object)) == [
        "+33123456789",
        "+33123456789",
        pd.NA,
        pd.NA,
    ]


def test_pick_first_supports_case_insensitive_matching():
    df = pd.DataFrame(
        {
            "SIREN": ["123456789", "987654321"],
            "Other": ["foo", "bar"],
        }
    )

    series = standardize._pick_first(df, ["siren", "nic"])
    assert series is df["SIREN"]


def test_merge_columns_prefers_non_empty_values():
    df = pd.DataFrame(
        {
            "primary": ["", "Valeur", pd.NA],
            "secondary": ["Fallback", "", "Valeur secondaire"],
        }
    )

    merged = standardize._merge_columns(df, ["primary", "secondary"])

    assert list(merged.astype(object)) == [
        "Fallback",
        "Valeur",
        "Valeur secondaire",
    ]
