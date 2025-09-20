#!/usr/bin/env python3
"""
Tests for the AddressProcessor utility.
"""

import pytest
import pandas as pd
from utils.address_processor import AddressProcessor


class TestAddressProcessor:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AddressProcessor()
    
    def test_init(self):
        """Test AddressProcessor initialization."""
        assert self.processor.postal_code_regex is not None
        assert len(self.processor.street_indicators) > 0
        assert len(self.processor.abbreviation_patterns) > 0
    
    def test_process_single_address_valid(self):
        """Test processing a valid address."""
        address = "123 Rue de la Paix 75001 Paris"
        result = self.processor._process_single_address(address)
        
        assert result['original'] == address
        assert result['cleaned'] != ""
        assert len(result['variants']) > 0
        assert result['quality_score'] > 0.5
        assert result['search_strategy'] in ['precise', 'standard', 'fuzzy', 'fallback']
    
    def test_process_single_address_empty(self):
        """Test processing an empty address."""
        result = self.processor._process_single_address("")
        
        assert result['original'] == ""
        assert result['cleaned'] == ""
        assert result['variants'] == []
        assert result['quality_score'] == 0.0
        assert result['search_strategy'] == 'skip'
    
    def test_process_single_address_none(self):
        """Test processing a None address."""
        result = self.processor._process_single_address(None)
        
        assert result['quality_score'] == 0.0
        assert result['search_strategy'] == 'skip'
    
    def test_clean_address(self):
        """Test address cleaning functionality."""
        address = "  123   RUE   DE LA PAIX   75001   PARIS  "
        cleaned = self.processor._clean_address(address)
        
        assert "  " not in cleaned  # No double spaces
        assert cleaned.startswith("123")
        assert "Rue" in cleaned  # Proper capitalization
    
    def test_generate_address_variants(self):
        """Test address variant generation."""
        address = "123 Rue de la Paix 75001 Paris"
        variants = self.processor._generate_address_variants(address)
        
        assert len(variants) > 0
        assert address in variants  # Original should be included
        
        # Should have variant without building number
        no_number_variant = next((v for v in variants if not v.startswith("123")), None)
        assert no_number_variant is not None
    
    def test_calculate_address_quality_high(self):
        """Test quality calculation for high quality address."""
        address = "123 Rue de la Paix 75001 Paris"
        quality = self.processor._calculate_address_quality(address)
        
        assert quality >= 0.7  # Should be high quality
    
    def test_calculate_address_quality_low(self):
        """Test quality calculation for low quality address."""
        address = "Paris"
        quality = self.processor._calculate_address_quality(address)
        
        assert quality <= 0.3  # Should be low quality
    
    def test_get_search_strategy(self):
        """Test search strategy determination."""
        assert self.processor._get_search_strategy(0.9) == 'precise'
        assert self.processor._get_search_strategy(0.6) == 'standard'
        assert self.processor._get_search_strategy(0.4) == 'fuzzy'
        assert self.processor._get_search_strategy(0.1) == 'fallback'
    
    def test_process_dataframe(self):
        """Test processing addresses in a DataFrame."""
        df = pd.DataFrame({
            'adresse': [
                "123 Rue de la Paix 75001 Paris",
                "456 Avenue des Champs 75008 Paris",
                "Invalid"
            ],
            'other_column': ['A', 'B', 'C']
        })
        
        result_df = self.processor.process_dataframe(df)
        
        # Check that original columns are preserved
        assert 'adresse' in result_df.columns
        assert 'other_column' in result_df.columns
        
        # Check that new columns are added
        assert 'address_cleaned' in result_df.columns
        assert 'address_variants' in result_df.columns
        assert 'address_primary' in result_df.columns
        assert 'address_quality_score' in result_df.columns
        assert 'address_search_strategy' in result_df.columns
        
        # Check data integrity
        assert len(result_df) == len(df)
        assert result_df['address_quality_score'].iloc[0] > result_df['address_quality_score'].iloc[2]
    
    def test_process_dataframe_missing_column(self):
        """Test processing DataFrame with missing address column."""
        df = pd.DataFrame({'other_column': ['A', 'B', 'C']})
        
        with pytest.raises(ValueError, match="Column 'adresse' not found"):
            self.processor.process_dataframe(df)
    
    def test_enhance_addresses(self):
        """Test enhancing a list of addresses."""
        addresses = [
            "123 Rue de la Paix 75001 Paris",
            "456 Avenue des Champs 75008 Paris",
            ""
        ]
        
        enhanced = self.processor.enhance_addresses(addresses)
        
        assert len(enhanced) == len(addresses)
        assert all('original' in addr for addr in enhanced)
        assert all('quality_score' in addr for addr in enhanced)
        assert enhanced[0]['quality_score'] > enhanced[2]['quality_score']
    
    @pytest.mark.parametrize("address,expected_min_quality", [
        ("123 Rue de la Paix 75001 Paris", 0.7),
        ("Rue de la Paix Paris", 0.3),
        ("Paris", 0.0),  # Single city name gets minimal score
        ("", 0.0)
    ])
    def test_quality_scoring_parametrized(self, address, expected_min_quality):
        """Test quality scoring with various address formats."""
        quality = self.processor._calculate_address_quality(address)
        assert quality >= expected_min_quality