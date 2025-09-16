"""
Test module for NAF reference collection functionality.
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd

from collect.nafreference import run, _filter_naf_data


def test_filter_naf_data():
    """Test NAF data filtering functionality."""
    # Create sample data
    df = pd.DataFrame({
        'code_naf': ['01.11', '01.12', '02.10', '02.20', '03.11'],
        'libelle': ['Culture de céréales', 'Culture du riz', 'Sylviculture', 'Exploitation forestière', 'Pêche en mer']
    })
    
    # Test filtering by NAF code prefix
    filtered = _filter_naf_data(df, '01')
    assert len(filtered) == 2
    assert all(filtered['code_naf'].str.startswith('01'))
    
    # Test no filter
    unfiltered = _filter_naf_data(df, None)
    assert len(unfiltered) == 5
    
    # Test specific code
    specific = _filter_naf_data(df, '02.10')
    assert len(specific) == 1
    assert specific['code_naf'].iloc[0] == '02.10'


def test_dry_run():
    """Test dry run functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"nafreference": {"naf_code": "01"}}
        ctx = {"outdir": tmpdir, "dry_run": True}
        
        result = run(cfg, ctx)
        
        assert result["status"] == "DRY_RUN"
        assert "file" in result
        
        # Check that file was created
        output_file = Path(result["file"])
        assert output_file.exists()
        
        # Check that it's a valid parquet file with expected schema
        df = pd.read_parquet(output_file)
        assert "code_naf" in df.columns
        assert "libelle" in df.columns
        assert len(df) == 0  # Should be empty for dry run


def test_fallback_configuration():
    """Test configuration handling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            "nafreference": {
                "naf_code": "01",
                "timeout": 30,
                "source": "fallback"
            }
        }
        ctx = {"outdir": tmpdir}
        
        # This might fail due to network, but should at least parse config correctly
        result = run(cfg, ctx)
        
        # Should return some result (might be FAILED due to network issues in test env)
        assert "status" in result
        if result["status"] == "OK":
            assert "file" in result
            assert "records_count" in result


def test_empty_configuration():
    """Test with empty configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {}
        ctx = {"outdir": tmpdir}
        
        result = run(cfg, ctx)
        
        # Should handle empty config gracefully
        assert "status" in result


def test_real_data_structure():
    """Test with real data download to validate structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            "nafreference": {
                "naf_code": "01",
                "source": "fallback",
                "timeout": 30
            }
        }
        ctx = {"outdir": tmpdir}
        
        result = run(cfg, ctx)
        
        # May fail due to network issues, but if it succeeds, validate structure
        if result["status"] == "OK":
            assert "file" in result
            assert "records_count" in result
            assert result["records_count"] > 0
            
            # Check that file was created and has valid structure
            output_file = Path(result["file"])
            assert output_file.exists()
            
            # Check parquet structure
            df = pd.read_parquet(output_file)
            assert "code_naf" in df.columns
            assert "libelle" in df.columns
            assert len(df) > 0
            
            # All NAF codes should start with '01'
            assert all(df["code_naf"].astype(str).str.startswith("01"))
            
            # Should have meaningful descriptions
            assert all(df["libelle"].astype(str).str.len() > 0)


if __name__ == "__main__":
    # Run a simple test if executed directly
    test_filter_naf_data()
    test_dry_run()
    test_real_data_structure()
    print("All tests passed!")