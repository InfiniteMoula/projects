import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pd = pytest.importorskip("pandas")

from normalize.standardize import run


@pytest.mark.parametrize("naf_code", ["6201Z", "62.01Z"])
def test_standardize_generates_summary_report(tmp_path, naf_code):
    df = pd.DataFrame(
        {
            "siren": ["123456789"],
            "siret": ["12345678900017"],
            "denominationUniteLegale": ["Test SARL"],
            "libelleCommuneEtablissement": ["Paris"],
            "codePostalEtablissement": ["75001"],
            "adresseEtablissement": ["1 rue de la Paix"],
            "activitePrincipaleEtablissement": ["62.01Z"],
            "dateCreationEtablissement": ["2020-01-01"],
            "telephone": ["0102030405"],
            "email": ["contact@example.com"],
            "siteweb": ["https://example.com"],
            "nomUniteLegale": ["Doe"],
            "prenomsUniteLegale": ["John"],
            "etatAdministratifEtablissement": ["A"],
        }
    )
    input_path = tmp_path / "input.parquet"
    df.to_parquet(input_path, index=False)

    outdir = tmp_path / "out"
    outdir.mkdir()
    log_path = tmp_path / "logs.jsonl"

    job = {
        "filters": {
            "naf_include": [naf_code],
            "active_only": True,
        },
        "kpi_targets": {"min_lines_per_s": 0.0},
    }
    ctx = {
        "input_path": input_path,
        "outdir_path": outdir,
        "outdir": str(outdir),
        "job": job,
        "logs": str(log_path),
    }

    result = run(job, ctx)

    assert result["status"] == "OK"
    assert result["rows"] == 1
    assert pytest.approx(result["rows_per_s"], rel=0.1) >= 0.0

    report_path = Path(result["report_path"])
    assert report_path.is_file()
    summary = json.loads(report_path.read_text(encoding="utf-8"))
    assert summary["rows_written"] == 1
    assert summary["batches"] == 1
    assert summary["files"]["csv"].endswith("normalized.csv")
    assert summary["kpi_evaluations"][0]["name"] == "lines_per_s"

    assert any(str(report_path) == path for path in result["files"])

    log_lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert any('"event": "summary"' in line for line in log_lines)
