from __future__ import annotations

import pandas as pd

from utils.filters import (
    PREMIUM_COLUMNS,
    filter_premium_columns,
    run_finalize_premium_dataset,
)


def test_filter_premium_columns_builds_address_and_dedupes(tmp_path):
    df = pd.DataFrame(
        {
            "denomination_usuelle": ["Alpha Conseil", "Alpha Conseil"],
            "numero_voie": [12, 12],
            "libelle_voie": ["Rue de Paris", "Rue de Paris"],
            "code_postal": ["75001", "75001"],
            "ville": ["Paris", "Paris"],
            "domain": ["https://alpha.fr", "alpha.fr"],
            "site": ["www.alpha.fr/contact", None],
            "siret": ["12345678900011", "12345678900011"],
            "siren": ["123456789", "123456789"],
            "effectif": ["12", "12"],
            "emails": ["sales@alpha.fr;info@alpha.fr", None],
            "telephone": [None, "+33 1 23 45 67 89"],
            "score_quality": [0.92, 0.91],
        }
    )

    filtered = filter_premium_columns(df)

    assert len(filtered) == 1, "Les doublons SIRET doivent être éliminés"
    row = filtered.iloc[0]

    assert row["adresse_complete"] == "12 Rue de Paris, 75001 Paris"
    assert row["domain"] == "alpha.fr"
    assert row["site_web"] == "https://alpha.fr/contact"
    assert row["employee_count"] == "11-50"
    assert row["email"] == "sales@alpha.fr"
    assert set(filtered.columns).issubset(set(PREMIUM_COLUMNS))


def test_filter_premium_columns_handles_missing_values():
    df = pd.DataFrame(
        {
            "denomination_usuelle": [None],
            "numero_voie": [None],
            "libelle_voie": [None],
            "code_postal": [None],
            "ville": [None],
            "domain": [None],
            "siret": [None],
        }
    )

    filtered = filter_premium_columns(df)

    assert "company_name" in filtered.columns
    assert "adresse_complete" in filtered.columns
    assert filtered.isna().all(axis=0).sum() >= 1  # colonnes sans données doivent rester sans erreur
    assert len(filtered.columns) <= len(PREMIUM_COLUMNS)


def test_filter_premium_columns_deduplicates_on_domain_when_siret_missing():
    df = pd.DataFrame(
        {
            "denomination_usuelle": ["Beta Labs", "Beta Labs"],
            "numero_voie": [5, 5],
            "libelle_voie": ["Rue Victor", "Rue Victor"],
            "code_postal": ["69001", "69001"],
            "ville": ["Lyon", "Lyon"],
            "domain": ["https://beta.com", "beta.com"],
            "siret": [None, None],
        }
    )

    filtered = filter_premium_columns(df)
    assert len(filtered) == 1, "La déduplication doit utiliser le domaine quand le SIRET est absent"
    assert filtered.iloc[0]["domain"] == "beta.com"


def test_run_finalize_premium_dataset_creates_trimmed_files(tmp_path):
    enriched = pd.DataFrame(
        {
            "denomination_usuelle": ["Gamma"],
            "numero_voie": [20],
            "libelle_voie": ["Boulevard Central"],
            "code_postal": ["33000"],
            "ville": ["Bordeaux"],
            "domain": ["gamma.io"],
            "siret": ["55566677788899"],
            "siren": ["555666777"],
            "effectif": ["55"],
            "emails": ["contact@gamma.io"],
            "telephone": ["+33 5 12 34 56 78"],
            "score_quality": [0.87],
        }
    )
    enriched_path = tmp_path / "dataset_enriched.parquet"
    enriched.to_parquet(enriched_path, index=False)

    ctx = {
        "outdir": str(tmp_path),
        "outdir_path": tmp_path,
        "logger": None,
    }

    result = run_finalize_premium_dataset({}, ctx)

    assert result["status"] == "OK"
    premium_csv = tmp_path / "dataset.csv"
    premium_parquet = tmp_path / "dataset.parquet"

    assert premium_csv.exists()
    assert premium_parquet.exists()

    premium_df = pd.read_csv(premium_csv)
    assert list(premium_df.columns) == list(result["columns"])
    assert "company_name" in premium_df.columns
    assert premium_df.iloc[0]["adresse_complete"] == "20 Boulevard Central, 33000 Bordeaux"
