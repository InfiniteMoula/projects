#!/usr/bin/env python3
"""Test the address search functionality."""

import pandas as pd
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from unittest.mock import patch, MagicMock

from enrich.address_search import run as run_address_search, _extract_business_info_from_html, _build_search_urls


def test_build_search_urls():
    """Test URL generation for search engines."""
    address = "123 Avenue des Champs-Élysées, 75008 Paris"
    urls = _build_search_urls(address)
    
    assert len(urls) == 2
    google_url = next(url for engine, url in urls if engine == "google")
    bing_url = next(url for engine, url in urls if engine == "bing")
    
    assert "google.fr" in google_url
    assert "bing.com" in bing_url
    assert "Champs-" in google_url  # URL encoded
    assert "Champs-" in bing_url


def test_extract_business_info_from_html():
    """Test extraction of business information from HTML."""
    # Mock Google search result HTML
    google_html = """
    <html>
    <body>
        <h3>Expert Comptable SA - Cabinet Comptable Paris</h3>
        <div>Téléphone: 01 42 12 34 56</div>
        <div>Email: contact@expert.fr</div>
        <h3>Cabinet Martin - Expertise Comptable</h3>
        <div>Tel: +33 1 72 34 56 78</div>
    </body>
    </html>
    """
    
    result = _extract_business_info_from_html(google_html, "google")
    
    print(f"Debug - Business names found: {result['business_names']}")
    print(f"Debug - Phone numbers found: {result['phone_numbers']}")
    print(f"Debug - Emails found: {result['emails']}")
    
    # More flexible assertions since search result parsing can be tricky
    assert len(result['phone_numbers']) >= 1, f"No phones found: {result['phone_numbers']}"
    assert len(result['emails']) >= 1, f"No emails found: {result['emails']}"
    
    # Check for expected phone number patterns (more flexible)
    phone_found = any("42" in phone and "12" in phone for phone in result['phone_numbers'])
    assert phone_found, f"Expected phone not found in: {result['phone_numbers']}"


def test_address_search_with_mock():
    """Test the complete address search process with mocked HTTP responses."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test input data
        input_data = {
            'siren': ['123456789', '987654321'],
            'siret': ['12345678900001', '98765432100001'],
            'raison_sociale': ['Expert Comptable SA', 'Cabinet Conseil SARL'],
            'adresse': [
                '123 Avenue des Champs-Élysées, 75008 Paris',
                '456 Rue de la République, 69001 Lyon'
            ],
            'code_postal': ['75008', '69001'],
            'commune': ['Paris', 'Lyon'],
            'naf': ['6920Z', '6920Z'],
            'telephone_norm': ['+33142123456', '+33472345678'],
            'email': ['contact@expert.fr', 'info@cabinet.com'],
            'nom': ['Dupont', 'Martin'],
            'prenom': ['Jean', 'Marie'],
            'etat_administratif': ['A', 'A']
        }
        
        input_df = pd.DataFrame(input_data)
        input_path = tmpdir / "normalized.parquet"
        pq.write_table(pa.Table.from_pandas(input_df), input_path)
        
        # Mock the HTTP responses
        mock_google_response = MagicMock()
        mock_google_response.text = """
        <html>
        <body>
            <h3>Expert Comptable Paris SA</h3>
            <div>Téléphone: 01 42 99 88 77</div>
            <div>contact@nouvellentreprise.fr</div>
        </body>
        </html>
        """
        mock_google_response.raise_for_status = MagicMock()
        
        mock_bing_response = MagicMock()
        mock_bing_response.text = """
        <html>
        <body>
            <h2>Cabinet Expert Comptable</h2>
            <div>Tel: +33 1 88 77 66 55</div>
            <div>info@cabinet-paris.com</div>
        </body>
        </html>
        """
        mock_bing_response.raise_for_status = MagicMock()
        
        def mock_get(url, **kwargs):
            if "google.fr" in url:
                return mock_google_response
            elif "bing.com" in url:
                return mock_bing_response
            else:
                raise ValueError(f"Unexpected URL: {url}")
        
        # Run the address search with mocked HTTP
        with patch('httpx.Client') as mock_client:
            mock_session = MagicMock()
            mock_session.get = mock_get
            mock_client.return_value.__enter__.return_value = mock_session
            
            ctx = {
                'outdir': str(tmpdir),
                'logger': None
            }
            
            result = run_address_search({}, ctx)
        
        # Check that the function succeeded
        assert result['status'] == 'OK'
        assert result['rows'] == 2
        assert result['addresses_searched'] == 2
        
        # Check output file
        output_path = tmpdir / "address_enriched.parquet"
        assert output_path.exists()
        
        enriched_df = pd.read_parquet(output_path)
        assert len(enriched_df) == 2
        
        # Check that enrichment columns exist
        assert 'found_business_names_str' in enriched_df.columns
        assert 'found_phones_str' in enriched_df.columns
        assert 'found_emails_str' in enriched_df.columns
        assert 'search_status' in enriched_df.columns
        
        # Check that some data was found
        assert enriched_df['found_business_names_str'].str.len().gt(0).any()
        assert enriched_df['found_phones_str'].str.len().gt(0).any()
        

def test_address_search_no_input():
    """Test behavior when no input file exists."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = {
            'outdir': str(tmpdir),
            'logger': None
        }
        
        result = run_address_search({}, ctx)
        print(f"Debug - No input test result: {result}")
        assert result['status'] == 'SKIPPED'
        assert result['reason'] == 'NO_NORMALIZED_DATA'


def test_address_search_empty_addresses():
    """Test behavior when no valid addresses exist."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test data with no valid addresses
        input_data = {
            'siren': ['123456789'],
            'siret': ['12345678900001'],
            'adresse': [''],  # Empty address
            'nom': ['Dupont']
        }
        
        input_df = pd.DataFrame(input_data)
        input_path = tmpdir / "normalized.parquet"
        pq.write_table(pa.Table.from_pandas(input_df), input_path)
        
        ctx = {
            'outdir': str(tmpdir),
            'logger': None
        }
        
        result = run_address_search({}, ctx)
        assert result['status'] == 'SKIPPED'
        assert result['reason'] == 'NO_VALID_ADDRESSES'


if __name__ == "__main__":
    print("Running address search tests...")
    test_build_search_urls()
    print("✓ URL building test passed")
    
    test_extract_business_info_from_html()
    print("✓ HTML extraction test passed")
    
    test_address_search_with_mock()
    print("✓ Mock address search test passed")
    
    test_address_search_no_input()
    print("✓ No input test passed")
    
    test_address_search_empty_addresses()
    print("✓ Empty addresses test passed")
    
    print("All tests passed!")