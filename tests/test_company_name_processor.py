#!/usr/bin/env python3
"""
Tests for the CompanyNameProcessor utility.
"""

import pytest
import pandas as pd
from utils.company_name_processor import CompanyNameProcessor


class TestCompanyNameProcessor:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = CompanyNameProcessor()
    
    def test_init(self):
        """Test CompanyNameProcessor initialization."""
        assert len(self.processor.legal_forms) > 0
        assert len(self.processor.common_suffixes) > 0
        assert len(self.processor.noise_words) > 0
    
    def test_optimize_for_linkedin_valid(self):
        """Test optimizing a valid company name."""
        company_name = "Acme Consulting SARL"
        result = self.processor.optimize_for_linkedin(company_name)
        
        assert 'variants' in result
        assert 'confidence' in result
        assert 'strategy' in result
        assert 'primary' in result
        
        assert len(result['variants']) > 0
        assert result['confidence'] > 0.0
        assert company_name in result['variants']  # Original should be included
    
    def test_optimize_for_linkedin_empty(self):
        """Test optimizing an empty company name."""
        result = self.processor.optimize_for_linkedin("")
        
        assert result['variants'] == []
        assert result['confidence'] == 0.0
        assert result['strategy'] == 'skip'
    
    def test_optimize_for_linkedin_none(self):
        """Test optimizing a None company name."""
        result = self.processor.optimize_for_linkedin(None)
        
        assert result['variants'] == []
        assert result['confidence'] == 0.0
        assert result['strategy'] == 'skip'
    
    def test_remove_legal_forms(self):
        """Test legal form removal."""
        test_cases = [
            ("Acme Company SARL", "Acme Company"),
            ("Tech Solutions SAS", "Tech Solutions"),
            ("Business (SA)", "Business"),
            ("Enterprise EURL", "Enterprise")
        ]
        
        for original, expected in test_cases:
            result = self.processor._remove_legal_forms(original)
            assert result == expected
    
    def test_remove_suffixes(self):
        """Test common suffix removal."""
        test_cases = [
            ("Acme Consulting", "Acme"),
            ("Tech Solutions", "Tech"),
            ("Business & Associés", "Business"),
            ("Company Services", "Company")
        ]
        
        for original, expected in test_cases:
            result = self.processor._remove_suffixes(original)
            assert result == expected
    
    def test_create_acronym(self):
        """Test acronym creation."""
        test_cases = [
            ("Acme Business Solutions", "ABS"),
            ("International Tech Company", "ITC"),
            ("Small Company", "SC"),  # Two words >= 2 chars each
            ("A", None)  # Single word
        ]
        
        for company_name, expected in test_cases:
            result = self.processor._create_acronym(company_name)
            assert result == expected
    
    def test_extract_core_name(self):
        """Test core name extraction."""
        test_cases = [
            ("Acme Business Solutions", "Acme"),
            ("International Technology Corporation", "International"),
            ("Small Co", "Small"),
            ("", None)
        ]
        
        for company_name, expected in test_cases:
            result = self.processor._extract_core_name(company_name)
            assert result == expected
    
    def test_remove_noise_words(self):
        """Test noise word removal."""
        company_name = "Société de la Grande Distribution"
        result = self.processor._remove_noise_words(company_name)
        
        # Should remove 'de' and 'la' but keep significant words
        # Note: 'de' might be part of 'Grande' so check words separately
        words = result.split()
        assert "de" not in words
        assert "la" not in words
        assert "Société" in result
        assert "Grande" in result
        assert "Distribution" in result
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        # High confidence case
        variants = ["Acme Company", "Acme", "AC", "Company"]
        confidence = self.processor._calculate_confidence("Acme Company SARL", variants)
        assert confidence >= 0.6
        
        # Low confidence case
        variants = ["A"]
        confidence = self.processor._calculate_confidence("A", variants)
        assert confidence <= 0.5
    
    def test_get_search_strategy(self):
        """Test search strategy determination."""
        variants = ["Company Name", "Company", "CN"]
        
        assert self.processor._get_search_strategy(0.9, variants) == 'comprehensive'
        assert self.processor._get_search_strategy(0.7, variants) == 'standard'
        assert self.processor._get_search_strategy(0.4, variants) == 'basic'
        assert self.processor._get_search_strategy(0.1, variants) == 'minimal'
    
    def test_process_dataframe(self):
        """Test processing company names in a DataFrame."""
        df = pd.DataFrame({
            'denomination': [
                "Acme Consulting SARL",
                "Tech Solutions SAS",
                "Small Co"
            ],
            'other_column': ['A', 'B', 'C']
        })
        
        result_df = self.processor.process_dataframe(df)
        
        # Check that original columns are preserved
        assert 'denomination' in result_df.columns
        assert 'other_column' in result_df.columns
        
        # Check that new columns are added
        assert 'company_variants' in result_df.columns
        assert 'company_primary' in result_df.columns
        assert 'company_confidence' in result_df.columns
        assert 'company_search_strategy' in result_df.columns
        
        # Check data integrity
        assert len(result_df) == len(df)
        assert result_df['company_confidence'].iloc[0] > 0
    
    def test_process_dataframe_alternative_columns(self):
        """Test processing DataFrame with alternative company name columns."""
        df = pd.DataFrame({
            'raison_sociale': [
                "Acme Consulting SARL",
                "Tech Solutions SAS"
            ]
        })
        
        result_df = self.processor.process_dataframe(df, 'raison_sociale')
        
        assert 'company_variants' in result_df.columns
        assert len(result_df) == len(df)
    
    def test_process_dataframe_missing_column(self):
        """Test processing DataFrame with missing company column."""
        df = pd.DataFrame({'other_column': ['A', 'B', 'C']})
        
        with pytest.raises(ValueError, match="No company name column found"):
            self.processor.process_dataframe(df)
    
    def test_get_search_terms(self):
        """Test getting prioritized search terms."""
        company_name = "Acme Business Consulting Solutions SARL"
        terms = self.processor.get_search_terms(company_name, max_terms=3)
        
        assert len(terms) <= 3
        assert len(terms) > 0
        
        # Should prioritize more complete names first
        assert len(terms[0].split()) >= len(terms[-1].split()) if len(terms) > 1 else True
    
    @pytest.mark.parametrize("company_name,expected_base_variants", [
        ("Acme SARL", ["Acme SARL", "Acme"]),
        ("Tech Solutions & Associés", ["Tech Solutions & Associés", "Tech"]),  # Simplified expectation
        ("Small", ["Small"])
    ])
    def test_variant_generation_parametrized(self, company_name, expected_base_variants):
        """Test variant generation with various company name formats."""
        result = self.processor.optimize_for_linkedin(company_name)
        variants = result['variants']
        
        # Check that key expected variants are present (not all, as algorithm may generate more)
        for expected in expected_base_variants:
            assert expected in variants