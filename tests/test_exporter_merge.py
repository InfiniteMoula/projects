from pathlib import Path

import pandas as pd

from package import exporter


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _run_export(outdir: Path) -> pd.DataFrame:
    result = exporter.run({}, {"outdir": str(outdir), "run_id": "test", "logger": None})
    assert result["status"] == "OK"
    parquet_path = outdir / "dataset_enriched.parquet"
    assert parquet_path.exists()
    return pd.read_parquet(parquet_path)


def test_exporter_merges_contacts_by_siren(tmp_path: Path):
    base = pd.DataFrame(
        [
            {
                "siren": "123456789",
                "name": "Acme",
                "naf": "62.01Z",
                "region": "IDF",
                "domain": "acme.com",
            }
        ]
    )
    contacts = pd.DataFrame(
        [
            {
                "siren": "123456789",
                "domain": "acme.com",
                "emails": [["info@acme.com"]],
                "phones": [["+33123456789"]],
            }
        ]
    )
    _write_parquet(tmp_path / "normalized.parquet", base)
    _write_parquet(tmp_path / "contacts" / "contacts_clean.parquet", contacts)

    enriched = _run_export(tmp_path)
    assert enriched.loc[0, "emails"] == ["info@acme.com"]
    assert enriched.loc[0, "phones"] == ["+33123456789"]


def test_exporter_uses_domain_fallback_when_siren_missing(tmp_path: Path):
    base = pd.DataFrame(
        [
            {
                "siren": "987654321",
                "name": "DomainCo",
                "naf": "70.22Z",
                "region": "ARA",
                "domain": "domainco.fr",
            }
        ]
    )
    contacts = pd.DataFrame(
        [
            {
                "siren": None,
                "domain": "domainco.fr",
                "emails": [["contact@domainco.fr"]],
                "phones": [["+33455667788"]],
            }
        ]
    )
    _write_parquet(tmp_path / "normalized.parquet", base)
    _write_parquet(tmp_path / "contacts" / "contacts_clean.parquet", contacts)

    enriched = _run_export(tmp_path)
    assert enriched.loc[0, "emails"] == ["contact@domainco.fr"]
    assert enriched.loc[0, "phones"] == ["+33455667788"]

