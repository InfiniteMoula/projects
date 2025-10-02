import pandas as pd
from pathlib import Path

import package.exporter as exporter


def test_exporter_outputs_curated_columns(tmp_path, monkeypatch):
    data = {
        "Nom entreprise": [None],
        "denomination_usuelle": ["ACME Conseil"],
        "Identifiant (SIREN/SIRET/DUNS)": [None],
        "siret": ["12345678900011"],
        "Forme juridique": ["SARL"],
        "Date de création": ["2010-01-15"],
        "Secteur (NAF/APE)": ["69.20Z"],
        "Adresse": [None],
        "Adresse complète": ["10 Rue Exemple, 75001 Paris"],
        "Région / Département": ["Île-de-France / Paris"],
        "Téléphone standard": ["+33 1 23 45 67 89"],
        "Email générique": ["contact@acme.fr"],
        "Site web": ["https://acme.fr"],
        "Effectif": ["12"],
        "CA": [None],
        "Chiffre d'affaires (fourchette)": ["0-2M€"],
        "Nom + fonction du dirigeant": ["Jane Doe - Gérante"],
        "Email pro vérifié du dirigeant": ["jane.doe@acme.fr"],
        "Technologies web": ["WordPress"],
        "Score de solvabilité": [0.82],
        "score_quality": [0.82],
        "Croissance CA (N vs N-1)": ["+8%"],
        "extra_column": ["should be dropped"],
    }
    df = pd.DataFrame(data)
    normalized_path = tmp_path / "normalized.parquet"
    df.to_parquet(normalized_path, index=False)

    job_path = tmp_path / "job.yaml"
    job_path.write_text("profile: internal\n")

    # Avoid invoking the PDF renderer during tests
    monkeypatch.setattr(exporter, "generate_pdf_report", lambda html_path: None)

    result = exporter.run({}, {
        "outdir_path": tmp_path,
        "run_id": "test-run",
        "job_path": job_path,
    })

    assert result["status"] == "OK"

    dataset_path = tmp_path / "dataset.parquet"
    assert dataset_path.exists()
    dataset = pd.read_parquet(dataset_path)

    assert list(dataset.columns) == exporter.CURATED_COLUMN_ORDER
    assert "extra_column" not in dataset.columns
    assert dataset.loc[0, "Nom entreprise"] == "ACME Conseil"
    assert dataset.loc[0, "Identifiant (SIREN/SIRET/DUNS)"] == "12345678900011"
    assert dataset.loc[0, "CA"] == "0-2M€"
