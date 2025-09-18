#!/usr/bin/env python3
"""
Test Apify agents integration.

This test validates that the Apify agents module can be imported
and the basic configuration validation works.
"""

import os
import sys
import tempfile
import pytest
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.apify_agents import run


def test_apify_agents_import():
    """Test that the apify_agents module can be imported."""
    from api import apify_agents
    assert hasattr(apify_agents, 'run')


def test_apify_disabled():
    """Test that Apify agents handle disabled configuration correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        
        cfg = {
            "apify": {
                "enabled": False
            }
        }
        
        ctx = {
            "outdir_path": outdir,
            "dry_run": False
        }
        
        result = run(cfg, ctx)
        
        assert result["status"] == "DISABLED"
        assert "apify_disabled.parquet" in result["file"]


def test_apify_dry_run():
    """Test that Apify agents handle dry run correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        
        cfg = {
            "apify": {
                "enabled": True
            }
        }
        
        ctx = {
            "outdir_path": outdir,
            "dry_run": True
        }
        
        result = run(cfg, ctx)
        
        assert result["status"] == "DRY_RUN"
        assert "apify_empty.parquet" in result["file"]


def test_apify_no_token():
    """Test that Apify agents handle missing API token correctly."""
    # Ensure no API token is set
    old_token = os.environ.get('APIFY_API_TOKEN')
    if 'APIFY_API_TOKEN' in os.environ:
        del os.environ['APIFY_API_TOKEN']
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            
            cfg = {
                "apify": {
                    "enabled": True
                }
            }
            
            ctx = {
                "outdir_path": outdir,
                "dry_run": False
            }
            
            result = run(cfg, ctx)
            
            assert result["status"] == "NO_TOKEN"
            assert "APIFY_API_TOKEN" in result["error"]
            assert "apify_no_token.parquet" in result["file"]
    
    finally:
        # Restore original token if it existed
        if old_token:
            os.environ['APIFY_API_TOKEN'] = old_token


def test_apify_no_input_data():
    """Test that Apify agents handle missing input data correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        
        cfg = {
            "apify": {
                "enabled": True
            }
        }
        
        ctx = {
            "outdir_path": outdir,
            "dry_run": False
        }
        
        # Set a fake token to pass token validation
        os.environ['APIFY_API_TOKEN'] = 'fake_token_for_testing'
        
        try:
            result = run(cfg, ctx)
            
            assert result["status"] == "NO_INPUT"
            assert "apify_no_input.parquet" in result["file"]
        
        finally:
            # Clean up fake token
            if 'APIFY_API_TOKEN' in os.environ:
                del os.environ['APIFY_API_TOKEN']


def test_apify_with_addresses():
    """Test that Apify agents process address data correctly (without API calls)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        
        # Create test address data
        test_data = pd.DataFrame({
            'index': [0, 1, 2],
            'adresse': [
                '123 Rue de Rivoli, Paris, 75001',
                '456 Avenue des Champs-Élysées, Paris, 75008',
                '789 Boulevard Saint-Germain, Paris, 75006'
            ],
            'company_name': ['Test Company 1', 'Test Company 2', 'Test Company 3'],
            'siren': ['123456789', '987654321', '456789123'],
            'siret': ['12345678900001', '98765432100001', '45678912300001']
        })
        
        # Save test database.csv
        database_path = outdir / "database.csv"
        test_data.to_csv(database_path, index=False)
        
        cfg = {
            "apify": {
                "enabled": True,
                "max_addresses": 2,  # Limit for testing
                "google_places": {"enabled": False},  # Disable to avoid API calls
                "google_maps_contacts": {"enabled": False},
                "linkedin_premium": {"enabled": False}
            }
        }
        
        ctx = {
            "outdir_path": outdir,
            "dry_run": False
        }
        
        # Set a fake token to pass token validation
        os.environ['APIFY_API_TOKEN'] = 'fake_token_for_testing'
        
        try:
            result = run(cfg, ctx)
            
            # Should succeed even with disabled scrapers
            assert result["status"] == "SUCCESS"
            assert result["addresses_processed"] == 2  # Limited by max_addresses
            assert "apify_enriched.parquet" in result["file"]
            
            # Check output file exists and has expected structure
            output_df = pd.read_parquet(result["file"])
            expected_columns = [
                'index', 'adresse', 'company_name', 'siren', 'siret',
                'apify_places_found', 'apify_business_names', 'apify_phones',
                'apify_emails', 'apify_websites', 'apify_ratings', 'apify_categories',
                'apify_executives', 'apify_linkedin_profiles'
            ]
            
            for col in expected_columns:
                assert col in output_df.columns, f"Missing column: {col}"
            
            assert len(output_df) == 2  # Should be limited to 2 addresses
        
        finally:
            # Clean up fake token
            if 'APIFY_API_TOKEN' in os.environ:
                del os.environ['APIFY_API_TOKEN']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])