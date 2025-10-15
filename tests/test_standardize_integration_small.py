from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pd = pytest.importorskip("pandas")

from normalize.standardize import run  # noqa: E402


def _build_mock_sirene_df(rows: int = 50) -> "pd.DataFrame":
    data = {
        "siren": [],
        "siret": [],
        "denominationUniteLegale": [],
        "libelleCommuneEtablissement": [],
        "codePostalEtablissement": [],
        "adresseEtablissement": [],
        "activitePrincipaleEtablissement": [],
        "etatAdministratifEtablissement": [],
        "telephone": [],
        "email": [],
        "siteweb": [],
        "dateCreationEtablissement": [],
        "nomUniteLegale": [],
        "prenomsUniteLegale": [],
    }

    for idx in range(rows):
        siren = f"{100000000 + idx:09d}"
        siret_suffix = f"{idx:05d}"
        siret = f"{siren}{siret_suffix}"[-14:]
        commune = "Paris" if idx % 2 == 0 else "Lyon"
        postal = "75001" if commune == "Paris" else "69002"
        tel_digits = f"0{(idx % 9) + 1}{idx:08d}"[:10]
        tel_grouped = " ".join(tel_digits[i : i + 2] for i in range(0, 10, 2))

        data["siren"].append(siren)
        data["siret"].append(siret)
        data["denominationUniteLegale"].append(f"Entreprise {idx}")
        data["libelleCommuneEtablissement"].append(commune)
        data["codePostalEtablissement"].append(postal)
        data["adresseEtablissement"].append(f"{idx + 1} rue Exemple")
        data["activitePrincipaleEtablissement"].append("62.01Z" if idx % 3 else "62.02A")
        data["etatAdministratifEtablissement"].append("A")
        data["telephone"].append(tel_grouped)
        data["email"].append(f"contact{idx}@example.com")
        data["siteweb"].append(f"https://example{idx}.com")
        data["dateCreationEtablissement"].append(f"201{idx % 10}-01-01")
        data["nomUniteLegale"].append(f"Nom {idx}")
        data["prenomsUniteLegale"].append(f"Prenom {idx}")

    return pd.DataFrame(data)


def test_standardize_pipeline_with_mock_sirene_sample(tmp_path):
    df = _build_mock_sirene_df()

    input_path = tmp_path / "sirene.parquet"
    df.to_parquet(input_path, index=False)

    outdir = tmp_path / "out"
    outdir.mkdir()

    ctx = {
        "input_path": input_path,
        "outdir_path": outdir,
        "job": {
            "filters": {"naf_include": ["62.0"], "active_only": True},
            "standardize_batch_rows": 20,
        },
    }

    result = run({}, ctx)

    assert result["status"] == "OK"
    assert result["rows"] == 50
    assert Path(result["report_path"]).is_file()
    assert (outdir / "normalized.parquet").exists()

    normalized_df = pd.read_parquet(outdir / "normalized.parquet")
    assert len(normalized_df) == 50

    assert set(normalized_df["departement"].unique()) == {"75", "69"}
    assert normalized_df["Téléphone standard"].notna().all()
    assert normalized_df["Nom entreprise"].iloc[0] == "Entreprise 0"
    assert normalized_df["Identifiant (SIREN/SIRET/DUNS)"].str.len().eq(14).all()

    summary = result["summary"]
    assert summary["rows_written"] == 50
    assert summary["batches"] >= 3  # 50 rows with batch size 20 ⇒ at least 3 batches
