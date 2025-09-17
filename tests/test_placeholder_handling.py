"""Tests for placeholder value handling in data validation."""

import pandas as pd
import pytest
import tempfile
from pathlib import Path

pytest.importorskip("pyarrow")

from enrich.phone_checks import run as run_phone_checks
from normalize.standardize import _is_placeholder, PLACEHOLDER_VALUES


def test_placeholder_identification():
    """Test that the _is_placeholder function correctly identifies placeholder values."""
    
    # Create test series with mixed values
    test_data = pd.Series([
        "TELEPHONE NON RENSEIGNE",
        "+33123456789", 
        "ADRESSE NON RENSEIGNEE",
        "Valid Company Name",
        "DENOMINATION NON RENSEIGNEE",
        "",
        None,
    ])
    
    result = _is_placeholder(test_data)
    
    # Check that placeholders are identified correctly
    assert result.iloc[0] == True  # TELEPHONE NON RENSEIGNE
    assert result.iloc[1] == False  # +33123456789 
    assert result.iloc[2] == True  # ADRESSE NON RENSEIGNEE
    assert result.iloc[3] == False  # Valid Company Name
    assert result.iloc[4] == True  # DENOMINATION NON RENSEIGNEE
    assert result.iloc[5] == False  # empty string
    assert result.iloc[6] == False  # None


def test_phone_validation_treats_placeholders_as_invalid():
    """Test that phone validation treats placeholder values as invalid."""
    
    # Create test data with valid phones and placeholders
    df = pd.DataFrame({
        "siren": ["123456789", "987654321", "111222333"],
        "siret": ["12345678900017", "98765432100024", "11122233300031"],
        "telephone_norm": [
            "+33123456789",  # Valid phone
            "TELEPHONE NON RENSEIGNE",  # Placeholder - should be invalid
            "+33987654321"   # Valid phone
        ],
        "raison_sociale": ["Company A", "Company B", "Company C"],
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        outdir = tmpdir / "out"
        outdir.mkdir()
        
        # Place input file in the output directory as expected by phone_checks
        input_path = outdir / "normalized.parquet"
        df.to_parquet(input_path, index=False)
        
        ctx = {
            "outdir_path": outdir,
            "outdir": str(outdir),
        }
        
        result = run_phone_checks({}, ctx)
        
        assert result["status"] == "OK"
        
        # Read the output to check phone validation
        output_path = outdir / "enriched_phone.parquet"
        assert output_path.exists()
        
        output_df = pd.read_parquet(output_path)
        
        # Check phone validation results
        assert output_df.loc[0, 'phone_valid'] == True   # Valid phone
        assert output_df.loc[1, 'phone_valid'] == False  # Placeholder should be invalid
        assert output_df.loc[2, 'phone_valid'] == True   # Valid phone


def test_all_placeholder_values_defined():
    """Test that all known placeholder values are included in the PLACEHOLDER_VALUES set."""
    
    # These are the placeholder values found in the codebase
    expected_placeholders = {
        "TELEPHONE NON RENSEIGNE",
        "ADRESSE NON RENSEIGNEE", 
        "DENOMINATION NON RENSEIGNEE"
    }
    
    assert PLACEHOLDER_VALUES == expected_placeholders


def test_data_quality_metrics_with_placeholders():
    """Test that data quality metrics can distinguish between real data and placeholders."""
    
    # Test data with mix of real and placeholder values
    test_data = pd.DataFrame({
        "telephone_norm": [
            "+33123456789",          # Real phone
            "TELEPHONE NON RENSEIGNE",  # Placeholder 
            "+33987654321",          # Real phone
            "TELEPHONE NON RENSEIGNE",  # Placeholder
        ],
        "adresse": [
            "123 rue de la Paix",    # Real address
            "ADRESSE NON RENSEIGNEE",   # Placeholder
            "456 avenue Victor Hugo", # Real address  
            "ADRESSE NON RENSEIGNEE",   # Placeholder
        ],
        "raison_sociale": [
            "Company ABC",           # Real name
            "Company XYZ",           # Real name
            "DENOMINATION NON RENSEIGNEE", # Placeholder
            "Company 123",           # Real name
        ]
    })
    
    # Calculate data quality metrics
    phone_real_data = ~_is_placeholder(test_data['telephone_norm'])
    address_real_data = ~_is_placeholder(test_data['adresse'])
    name_real_data = ~_is_placeholder(test_data['raison_sociale'])
    
    # Check that placeholders are correctly identified as missing data
    assert phone_real_data.sum() == 2    # 2 real phones, 2 placeholders
    assert address_real_data.sum() == 2  # 2 real addresses, 2 placeholders  
    assert name_real_data.sum() == 3     # 3 real names, 1 placeholder
    
    # Calculate completion rates excluding placeholders
    phone_completion_rate = phone_real_data.mean()
    address_completion_rate = address_real_data.mean()
    name_completion_rate = name_real_data.mean()
    
    assert phone_completion_rate == 0.5   # 50% real data
    assert address_completion_rate == 0.5 # 50% real data
    assert name_completion_rate == 0.75   # 75% real data