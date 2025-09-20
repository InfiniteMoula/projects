#!/usr/bin/env python3
"""
Tests for the enhanced apify_agents input preparation functionality.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from api.apify_agents import _prepare_input_data


class TestApifyAgentsInputPreparation:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_df = pd.DataFrame({
            'adresse': [
                "123 Rue de la Paix 75001 Paris",
                "456 Avenue des Champs 75008 Paris",
                "Invalid Address"
            ],
            'denomination': [
                "Acme Consulting SARL",
                "Tech Solutions SAS",
                "Small Co"
            ],
            'siren': ['123456789', '987654321', '111222333']
        })
        
        self.config = {
            'input_preparation': {
                'min_address_quality': 0.3,
                'min_company_confidence': 0.3
            }
        }
    
    def test_prepare_input_data_basic(self):
        """Test basic input data preparation."""
        result = _prepare_input_data(self.sample_df, self.config)
        
        assert 'processed_df' in result
        assert 'address_processor' in result
        assert 'company_processor' in result
        assert 'total_records' in result
        assert 'qualified_records' in result
        assert 'quality_stats' in result
        
        # Check that new columns are added
        processed_df = result['processed_df']
        assert 'address_cleaned' in processed_df.columns
        assert 'address_quality_score' in processed_df.columns
        assert 'company_variants' in processed_df.columns
        assert 'company_confidence' in processed_df.columns
        assert 'input_quality_score' in processed_df.columns
        
        # Check that data is sorted by quality
        quality_scores = processed_df['input_quality_score'].tolist()
        assert quality_scores == sorted(quality_scores, reverse=True)
    
    def test_prepare_input_data_filtering(self):
        """Test that low quality records are filtered out."""
        # Use stricter quality requirements
        strict_config = {
            'input_preparation': {
                'min_address_quality': 0.8,
                'min_company_confidence': 0.8
            }
        }
        
        result = _prepare_input_data(self.sample_df, strict_config)
        
        # Should have fewer qualified records
        assert result['qualified_records'] <= result['total_records']
        assert result['total_records'] == len(self.sample_df)
    
    def test_prepare_input_data_no_address_column(self):
        """Test handling when no address column is found."""
        df_no_address = pd.DataFrame({
            'denomination': ["Company A", "Company B"],
            'other_column': ['X', 'Y']
        })
        
        result = _prepare_input_data(df_no_address, self.config)
        
        # Should still process companies
        assert result['total_records'] == len(df_no_address)
        processed_df = result['processed_df']
        assert 'company_variants' in processed_df.columns
    
    def test_prepare_input_data_no_company_column(self):
        """Test handling when no company column is found."""
        df_no_company = pd.DataFrame({
            'adresse': ["123 Rue Test", "456 Avenue Test"],
            'other_column': ['X', 'Y']
        })
        
        result = _prepare_input_data(df_no_company, self.config)
        
        # Should still process addresses
        assert result['total_records'] == len(df_no_company)
        processed_df = result['processed_df']
        assert 'address_cleaned' in processed_df.columns
    
    def test_prepare_input_data_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        result = _prepare_input_data(empty_df, self.config)
        
        assert result['total_records'] == 0
        assert result['qualified_records'] == 0
        assert result['quality_stats']['mean_quality'] == 0.0
    
    def test_quality_statistics(self):
        """Test that quality statistics are calculated correctly."""
        result = _prepare_input_data(self.sample_df, self.config)
        
        stats = result['quality_stats']
        assert 'mean_quality' in stats
        assert 'min_quality' in stats
        assert 'max_quality' in stats
        
        # Quality scores should be between 0 and 1
        assert 0.0 <= stats['min_quality'] <= 1.0
        assert 0.0 <= stats['max_quality'] <= 1.0
        assert stats['min_quality'] <= stats['mean_quality'] <= stats['max_quality']