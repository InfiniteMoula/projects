from pathlib import Path

import json
import pandas as pd

from marketing_exports import generate_marketing_exports
import export_marketing


def _sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "siren": "111111111",
                "name": "Alpha Conseil",
                "email": "sales@alpha.fr",
                "email_valid": True,
                "telephone": "+33123456789",
                "telephone_valid": True,
                "naf": "62.01Z",
                "region": "Île-de-France",
                "score_business": 95.0,
            },
            {
                "siren": "222222222",
                "name": "Beta Industrie",
                "email": "",
                "email_valid": False,
                "telephone": "+33987654321",
                "telephone_valid": True,
                "naf": "70.22Z",
                "region": "Auvergne-Rhône-Alpes",
                "score_business": 82.5,
            },
            {
                "siren": "333333333",
                "name": "Gamma SARL",
                "email": "gamma@example.com",
                "email_valid": True,
                "telephone": "",
                "telephone_valid": False,
                "naf": "70.22Z",
                "region": "Occitanie",
                "score_business": 40.0,
            },
            {
                "siren": "222222222",
                "name": "Beta Industrie Duplicate",
                "email": "contact@beta.fr",
                "email_valid": True,
                "telephone": "",
                "telephone_valid": False,
                "naf": "69.10Z",
                "region": "Auvergne-Rhône-Alpes",
                "score_business": 60.0,
            },
            {
                "siren": "444444444",
                "name": "Delta SAS",
                "email": "",
                "email_valid": False,
                "telephone": "",
                "telephone_valid": False,
                "naf": "62.01Z",
                "region": "Île-de-France",
                "score_business": 20.0,
            },
        ]
    )


def test_generate_marketing_exports(tmp_path):
    dataset = _sample_dataset()
    dataset_path = tmp_path / "dataset_enriched.parquet"
    dataset.to_parquet(dataset_path, index=False)

    summary = generate_marketing_exports(outdir=tmp_path, limit=2)

    export_dir = tmp_path / "marketing_exports"
    csv_path = export_dir / "top_contactables_2.csv"
    parquet_path = export_dir / "top_contactables_2.parquet"
    dashboard_path = export_dir / "dashboard.html"

    assert csv_path.exists()
    assert parquet_path.exists()
    assert dashboard_path.exists()

    exported = pd.read_csv(csv_path)
    # Only two best contactable companies remain after deduplication
    assert len(exported) == 2
    assert exported.loc[0, "company_name"] == "Alpha Conseil"
    assert bool(exported.loc[0, "email_valid"])

    html_content = dashboard_path.read_text(encoding="utf-8")
    assert "Total leads" in html_content
    assert "Top 5 codes NAF" in html_content
    assert "62.01Z" in html_content

    assert summary["total_leads"] == len(dataset)
    assert summary["contactable_leads"] == 4  # Delta SAS is not contactable


def test_export_marketing_cli(tmp_path, monkeypatch, capsys):
    dataset = _sample_dataset()
    dataset_path = tmp_path / "dataset_enriched.parquet"
    dataset.to_parquet(dataset_path, index=False)

    result = export_marketing.main(
        ["--outdir", str(tmp_path), "--limit", "1"],
    )
    assert result["contactable_leads"] == 4

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["top_contactables_csv"].endswith("top_contactables_1.csv")

    assert (tmp_path / "marketing_exports" / "top_contactables_1.csv").exists()
