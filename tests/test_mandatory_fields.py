"""Tests for data preservation and fallback logic in standardize step."""

import pandas as pd
import pytest
import tempfile
from pathlib import Path

pytest.importorskip("pyarrow")

from normalize.standardize import run


def test_data_preservation_and_fallback_logic():
    """Test that genuine data is preserved and fallback logic works without placeholder pollution."""
    
    # Create test data with various missing field scenarios
    df = pd.DataFrame({
        "siren": ["123456789", "987654321", "111222333"],
        "siret": ["12345678900017", "98765432100024", "11122233300031"],
        
        # Test raison_sociale fallback
        "denominationUniteLegale": ["Company A", "", ""],  # Missing for rows 1,2
        "denominationUsuelleEtablissement": ["", "Fallback B", ""],  # Fallback for row 1
        
        # Test enseigne fallback  
        "enseigne1Etablissement": ["Enseigne A", "", ""],  # Missing for rows 1,2
        
        "libelleCommuneEtablissement": ["Paris", "Lyon", "Marseille"],
        "codePostalEtablissement": ["75001", "69000", "13000"],
        
        # Test adresse preservation vs null
        "adresseEtablissement": ["1 rue Test", "", ""],  # Missing for rows 1,2
        
        "activitePrincipaleEtablissement": ["62.01Z", "43.29A", "69.20Z"],
        "dateCreationEtablissement": ["2020-01-01", "2019-05-15", "2021-03-10"],
        
        # Test telephone preservation vs null
        "telephone": ["0102030405", "", ""],  # Missing for rows 1,2
        
        "email": ["contact@a.com", "", ""],
        "siteweb": ["https://a.com", "", ""],
        "nomUniteLegale": ["Dupont", "Martin", "Dubois"],
        "prenomsUniteLegale": ["Jean", "Paul", "Marie"],
        "etatAdministratifEtablissement": ["A", "A", "A"],
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "input.parquet"
        df.to_parquet(input_path, index=False)
        
        outdir = tmpdir / "out"
        outdir.mkdir()
        
        job = {
            "filters": {
                "naf_include": ["6201Z", "4329A", "6920Z"],
                "active_only": True,
            },
        }
        ctx = {
            "input_path": input_path,
            "outdir_path": outdir,
            "outdir": str(outdir),
            "job": job,
        }
        
        result = run(job, ctx)
        
        assert result["status"] == "OK"
        assert result["rows"] == 3
        
        # Read the output CSV to validate data preservation  
        csv_path = outdir / "normalized.csv"
        assert csv_path.exists()
        
        # Read CSV with string dtype to preserve phone number formatting
        output_df = pd.read_csv(csv_path, dtype={'telephone_norm': 'string'})
        
        # Check that core business fields with fallback logic are populated
        assert output_df.loc[0, 'raison_sociale'] == 'Company A'
        assert output_df.loc[1, 'raison_sociale'] == 'Fallback B'  # Falls back to denominationUsuelleEtablissement
        assert pd.isna(output_df.loc[2, 'raison_sociale'])  # No data available, should be null
        
        assert output_df.loc[0, 'enseigne'] == 'Enseigne A'
        assert output_df.loc[1, 'enseigne'] == 'Fallback B'  # Falls back to raison_sociale
        assert pd.isna(output_df.loc[2, 'enseigne'])  # Falls back to raison_sociale which is also null
        
        # Check that optional fields preserve null state when missing
        assert output_df.loc[0, 'adresse'] == '1 rue Test'
        assert pd.isna(output_df.loc[1, 'adresse'])  # Missing data preserved as null
        assert pd.isna(output_df.loc[2, 'adresse'])  # Missing data preserved as null
        
        assert output_df.loc[0, 'telephone_norm'] == '+33102030405'
        assert pd.isna(output_df.loc[1, 'telephone_norm'])  # Missing data preserved as null
        assert pd.isna(output_df.loc[2, 'telephone_norm'])  # Missing data preserved as null


def test_raison_sociale_fallback_priority():
    """Test that raison_sociale follows the correct fallback priority."""
    
    df = pd.DataFrame({
        "siren": ["111", "222", "333"],
        "denominationUniteLegale": ["Primary Name", "", ""],
        "denominationUsuelleEtablissement": ["Secondary Name", "Fallback Name", ""],
        "activitePrincipaleEtablissement": ["62.01Z", "62.01Z", "62.01Z"],
        "etatAdministratifEtablissement": ["A", "A", "A"],
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "input.parquet"
        df.to_parquet(input_path, index=False)
        
        outdir = tmpdir / "out"
        outdir.mkdir()
        
        job = {"filters": {"naf_include": ["6201Z"], "active_only": True}}
        ctx = {"input_path": input_path, "outdir_path": outdir, "outdir": str(outdir), "job": job}
        
        result = run(job, ctx)
        assert result["status"] == "OK"
        
        output_df = pd.read_csv(outdir / "normalized.csv", dtype={'telephone_norm': 'string'})
        
        # Row 0: Uses primary name
        assert output_df.loc[0, 'raison_sociale'] == 'Primary Name'
        
        # Row 1: Falls back to secondary name
        assert output_df.loc[1, 'raison_sociale'] == 'Fallback Name'
        
        # Row 2: No data available, should be null instead of placeholder
        assert pd.isna(output_df.loc[2, 'raison_sociale'])