"""Tests for mandatory field population in standardize step."""

import pandas as pd
import pytest
import tempfile
from pathlib import Path

pytest.importorskip("pyarrow")

from normalize.standardize import run


def test_mandatory_fields_always_populated():
    """Test that mandatory fields (raison_sociale, enseigne, adresse, telephone_norm) are always populated."""
    
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
        
        # Test adresse fallback
        "adresseEtablissement": ["1 rue Test", "", ""],  # Missing for rows 1,2
        
        "activitePrincipaleEtablissement": ["62.01Z", "43.29A", "69.20Z"],
        "dateCreationEtablissement": ["2020-01-01", "2019-05-15", "2021-03-10"],
        
        # Test telephone fallback
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
        
        # Read the output CSV to validate mandatory fields
        csv_path = outdir / "normalized.csv"
        assert csv_path.exists()
        
        output_df = pd.read_csv(csv_path)
        
        # Check each mandatory field is populated for all rows
        mandatory_fields = ['raison_sociale', 'enseigne', 'adresse', 'telephone_norm']
        
        for field in mandatory_fields:
            assert field in output_df.columns, f"Field {field} missing from output"
            
            for i, value in enumerate(output_df[field]):
                # Convert to string and check it's not empty/null
                str_value = str(value).strip()
                assert str_value not in ['', 'nan', 'None', '<NA>'], \
                    f"Row {i}: {field} is empty/null: '{value}'"
                assert pd.notna(value), f"Row {i}: {field} is NaN: '{value}'"
        
        # Test specific fallback behaviors
        assert output_df.loc[0, 'raison_sociale'] == 'Company A'
        assert output_df.loc[1, 'raison_sociale'] == 'Fallback B'  # Falls back to denominationUsuelleEtablissement
        assert output_df.loc[2, 'raison_sociale'] == 'DENOMINATION NON RENSEIGNEE'  # Final fallback
        
        assert output_df.loc[0, 'enseigne'] == 'Enseigne A'
        assert output_df.loc[1, 'enseigne'] == 'Fallback B'  # Falls back to raison_sociale
        assert output_df.loc[2, 'enseigne'] == 'DENOMINATION NON RENSEIGNEE'  # Falls back to raison_sociale
        
        assert output_df.loc[0, 'adresse'] == '1 rue Test'
        assert output_df.loc[1, 'adresse'] == 'ADRESSE NON RENSEIGNEE'  # Fallback placeholder
        assert output_df.loc[2, 'adresse'] == 'ADRESSE NON RENSEIGNEE'  # Fallback placeholder
        
        assert output_df.loc[0, 'telephone_norm'] == '+33102030405'
        assert output_df.loc[1, 'telephone_norm'] == 'TELEPHONE NON RENSEIGNE'  # Fallback placeholder
        assert output_df.loc[2, 'telephone_norm'] == 'TELEPHONE NON RENSEIGNE'  # Fallback placeholder


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
        
        output_df = pd.read_csv(outdir / "normalized.csv")
        
        # Row 0: Uses primary name
        assert output_df.loc[0, 'raison_sociale'] == 'Primary Name'
        
        # Row 1: Falls back to secondary name
        assert output_df.loc[1, 'raison_sociale'] == 'Fallback Name'
        
        # Row 2: Uses final fallback
        assert output_df.loc[2, 'raison_sociale'] == 'DENOMINATION NON RENSEIGNEE'