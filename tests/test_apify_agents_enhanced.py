#!/usr/bin/env python3
"""
Test the enhanced apify_agents.py functionality without requiring actual API calls.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from api.apify_agents import run

class TestApifyAgentsEnhanced:
    
    def test_apify_agents_disabled(self, tmp_path):
        """Test apify_agents when disabled in config."""
        # Create test data
        df = pd.DataFrame({
            'adresse': ["123 Rue Test", "456 Avenue Test"],
            'denomination': ["Company A", "Company B"]
        })
        
        # Save test data
        input_path = tmp_path / "normalized.parquet"
        df.to_parquet(input_path)
        
        # Configuration with apify disabled
        cfg = {
            'apify': {
                'enabled': False
            }
        }
        
        ctx = {
            'outdir': str(tmp_path),
            'outdir_path': tmp_path
        }
        
        result = run(cfg, ctx)
        
        assert result['status'] == 'DISABLED'
        assert 'file' in result
        
        # Check that disabled file was created
        disabled_file = tmp_path / "apify_disabled.parquet"
        assert disabled_file.exists()
    
    def test_apify_agents_no_token(self, tmp_path):
        """Test apify_agents when no API token is provided."""
        # Create test data
        df = pd.DataFrame({
            'adresse': ["123 Rue Test"],
            'denomination': ["Company A"]
        })
        
        input_path = tmp_path / "normalized.parquet"
        df.to_parquet(input_path)
        
        # Configuration with apify enabled but no token
        cfg = {
            'apify': {
                'enabled': True
            }
        }
        
        ctx = {
            'outdir': str(tmp_path),
            'outdir_path': tmp_path
        }
        
        # Ensure no token is set
        with patch.dict(os.environ, {}, clear=True):
            result = run(cfg, ctx)
        
        assert result['status'] == 'NO_TOKEN'
        assert 'error' in result
    
    def test_apify_agents_no_input_data(self, tmp_path):
        """Test apify_agents when no input data is available."""
        cfg = {
            'apify': {
                'enabled': True
            }
        }
        
        ctx = {
            'outdir': str(tmp_path),
            'outdir_path': tmp_path
        }
        
        with patch.dict(os.environ, {'APIFY_API_TOKEN': 'fake_token'}):
            result = run(cfg, ctx)
        
        assert result['status'] == 'NO_INPUT'
    
    @patch('api.apify_agents._get_apify_client')
    @patch('api.apify_agents._run_google_places_crawler')
    @patch('api.apify_agents._run_google_maps_contact_details')
    @patch('api.apify_agents._run_linkedin_premium_actor')
    def test_apify_agents_with_input_preparation(
        self, 
        mock_linkedin, 
        mock_contacts, 
        mock_places, 
        mock_client,
        tmp_path
    ):
        """Test apify_agents with input preparation and mock API calls."""
        
        # Setup mocks
        mock_client.return_value = MagicMock()
        mock_places.return_value = []
        mock_contacts.return_value = []
        mock_linkedin.return_value = []
        
        # Create test data with varying quality
        df = pd.DataFrame({
            'adresse': [
                "123 Rue de la Paix 75001 Paris",  # High quality
                "456 Avenue Test 75002 Paris",     # Medium quality  
                "Invalid Address"                   # Low quality
            ],
            'denomination': [
                "Acme Consulting SARL",             # High quality
                "Tech Solutions",                   # Medium quality
                "X"                                 # Low quality
            ],
            'siren': ['123456789', '234567890', '345678901']
        })
        
        input_path = tmp_path / "normalized.parquet"
        df.to_parquet(input_path)
        
        # Configuration with input preparation settings
        cfg = {
            'apify': {
                'enabled': True,
                'max_addresses': 10,
                'google_places': {'enabled': False},  # Disable to avoid actual calls
                'google_maps_contacts': {'enabled': False},
                'linkedin_premium': {'enabled': False}
            },
            'input_preparation': {
                'min_address_quality': 0.5,
                'min_company_confidence': 0.5
            }
        }
        
        ctx = {
            'outdir': str(tmp_path),
            'outdir_path': tmp_path
        }
        
        with patch.dict(os.environ, {'APIFY_API_TOKEN': 'fake_token'}):
            result = run(cfg, ctx)
        
        assert result['status'] == 'SUCCESS'
        assert 'input_quality_stats' in result
        assert 'qualification_rate' in result
        
        # Check that output file was created
        output_file = Path(result['file'])
        assert output_file.exists()
        
        # Load and verify the output data
        output_df = pd.read_parquet(output_file)
        
        # Should have enhanced columns from input preparation
        expected_columns = [
            'address_cleaned', 'address_quality_score', 'address_search_strategy',
            'company_primary', 'company_confidence', 'company_search_strategy',
            'input_quality_score'
        ]
        
        for col in expected_columns:
            assert col in output_df.columns, f"Missing expected column: {col}"
        
        # Should be sorted by quality (best first)
        quality_scores = output_df['input_quality_score'].tolist()
        assert quality_scores == sorted(quality_scores, reverse=True)
        
        print(f"✅ Enhanced apify_agents test passed!")
        print(f"   - Processed {len(output_df)} records")
        print(f"   - Qualification rate: {result.get('qualification_rate', 0):.2%}")
        print(f"   - Quality stats: {result.get('input_quality_stats', {})}")
        
        return True

if __name__ == "__main__":
    # Run a quick test
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_instance = TestApifyAgentsEnhanced()
        test_instance.test_apify_agents_with_input_preparation(tmp_dir)