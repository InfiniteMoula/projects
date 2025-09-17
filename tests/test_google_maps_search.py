"""Tests for Google Maps search enrichment module."""

import pytest
import tempfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from unittest.mock import Mock, patch

import enrich.google_maps_search as google_maps_search


@pytest.fixture
def temp_outdir():
    """Create a temporary output directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test normalized.csv data
        test_data = {
            'siren': ['123456789', '987654321', '555666777'],
            'siret': ['12345678900001', '98765432100001', '55566677700001'],
            'raison_sociale': ['Expert Comptable SA', 'Cabinet Conseil SARL', 'Avocat & Associés'],
            'adresse': ['123 Avenue des Champs-Élysées', '456 Rue de la République', '789 Boulevard de la Croisette'],
            'code_postal': ['75008', '69001', '06400'],
            'commune': ['Paris', 'Lyon', 'Cannes'],
            'naf': ['6920Z', '6920Z', '6910Z']
        }
        
        df = pd.DataFrame(test_data)
        
        # Save as parquet
        parquet_path = tmpdir / "normalized.parquet"
        pq.write_table(pa.Table.from_pandas(df), parquet_path)
        
        yield tmpdir


def test_build_address_query():
    """Test address query building from DataFrame row."""
    # Test with complete address components
    row_complete = pd.Series({
        'numero_voie': '123',
        'type_voie': 'Avenue',
        'libelle_voie': 'des Champs-Élysées',
        'commune': 'Paris',
        'code_postal': '75008',
        'raison_sociale': 'Expert Comptable SA'
    })
    
    query = google_maps_search._build_address_query(row_complete)
    expected = "Expert Comptable SA 123 Avenue des Champs-Élysées Paris 75008"
    assert query == expected
    
    # Test with only basic address
    row_basic = pd.Series({
        'adresse': '456 Rue de la République',
        'commune': 'Lyon',
        'code_postal': '69001',
        'raison_sociale': 'Cabinet Conseil SARL'
    })
    
    query = google_maps_search._build_address_query(row_basic)
    expected = "Cabinet Conseil SARL 456 Rue de la République Lyon 69001"
    assert query == expected


def test_build_google_maps_url():
    """Test Google Maps URL building."""
    query = "Expert Comptable SA 123 Avenue des Champs-Élysées Paris 75008"
    url = google_maps_search._build_google_maps_url(query)
    
    assert url.startswith("https://maps.google.com/maps/search/")
    assert "Expert+Comptable+SA" in url


def test_extract_business_info_from_maps():
    """Test business information extraction from mock HTML."""
    # Mock HTML content with business information
    mock_html = """
    <html>
        <body>
            <h1>Expert Comptable & Associés</h1>
            <div>Téléphone: 01 42 12 34 56</div>
            <div>Email: contact@expert.fr</div>
            <div>4,5 étoiles (120 avis)</div>
            <div>Cabinet d'expertise comptable</div>
            <div>Site web: https://www.expert-comptable.fr</div>
        </body>
    </html>
    """
    
    result = google_maps_search._extract_business_info_from_maps(mock_html)
    
    # Check that information was extracted
    assert len(result['phone_numbers']) > 0
    assert len(result['emails']) > 0
    assert len(result['business_names']) > 0
    
    # Verify specific extractions
    assert any('01 42 12 34 56' in phone for phone in result['phone_numbers'])
    assert any('contact@expert.fr' in email for email in result['emails'])


def test_merge_maps_results():
    """Test merging of Google Maps search results."""
    search_results = [
        {
            'query': 'Expert Comptable SA Paris',
            'business_names': ['Expert Comptable & Associés'],
            'phone_numbers': ['01 42 12 34 56'],
            'emails': ['contact@expert.fr'],
            'ratings': [4.5],
            'review_counts': [120],
            'business_types': ['Cabinet'],
            'websites': ['https://www.expert.fr'],
            'search_status': 'success'
        },
        {
            'query': 'Cabinet Conseil SARL Lyon',
            'business_names': ['Cabinet Conseil'],
            'phone_numbers': ['04 72 34 56 78'],
            'emails': [],
            'ratings': [],
            'review_counts': [],
            'business_types': ['Cabinet'],
            'websites': [],
            'search_status': 'success'
        }
    ]
    
    df = google_maps_search._merge_maps_results(search_results)
    
    assert len(df) == 2
    assert 'business_names_str' in df.columns
    assert 'phone_numbers_str' in df.columns
    assert 'rating' in df.columns
    assert 'review_count' in df.columns
    
    # Check string conversion
    assert df['business_names_str'].iloc[0] == 'Expert Comptable & Associés'
    assert df['phone_numbers_str'].iloc[0] == '01 42 12 34 56'
    assert df['rating'].iloc[0] == 4.5
    assert df['review_count'].iloc[0] == 120


@patch('enrich.google_maps_search.httpx.Client')
def test_run_google_maps_enrichment(mock_client, temp_outdir):
    """Test the complete Google Maps enrichment run."""
    # Mock HTTP client
    mock_session = Mock()
    mock_client.return_value.__enter__.return_value = mock_session
    
    # Mock response with business information
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = """
    <html>
        <body>
            <h1>Expert Comptable SA</h1>
            <div>01 42 12 34 56</div>
            <div>contact@expert.fr</div>
            <div>4,2 étoiles (85 avis)</div>
        </body>
    </html>
    """
    mock_session.get.return_value = mock_response
    
    # Setup context
    ctx = {
        "outdir": str(temp_outdir),
        "logger": None
    }
    
    # Run the enrichment
    result = google_maps_search.run({}, ctx)
    
    # Check result
    assert result['status'] == 'OK'
    assert result['records_processed'] == 3
    assert result['searches_performed'] > 0
    
    # Check output file
    output_path = temp_outdir / "google_maps_enriched.parquet"
    assert output_path.exists()
    
    # Verify enriched data
    enriched_df = pd.read_parquet(output_path)
    assert len(enriched_df) == 3
    assert 'maps_business_names' in enriched_df.columns
    assert 'maps_phone_numbers' in enriched_df.columns
    assert 'maps_emails' in enriched_df.columns
    assert 'maps_rating' in enriched_df.columns
    assert 'maps_search_status' in enriched_df.columns


def test_run_with_missing_data(temp_outdir):
    """Test behavior when normalized data is missing."""
    # Remove the normalized.parquet file
    (temp_outdir / "normalized.parquet").unlink()
    
    # Temporarily move the project root normalized.csv if it exists
    project_normalized = Path("/home/runner/work/projects/projects/normalized.csv")
    backup_path = None
    if project_normalized.exists():
        backup_path = Path("/tmp/normalized.csv.backup")
        project_normalized.rename(backup_path)
    
    try:
        ctx = {
            "outdir": str(temp_outdir),
            "logger": None
        }
        
        result = google_maps_search.run({}, ctx)
        
        assert result['status'] == 'SKIPPED'
        assert 'NO_NORMALIZED_DATA' in result['reason']
    finally:
        # Restore the file if it was moved
        if backup_path and backup_path.exists():
            backup_path.rename(project_normalized)


def test_run_with_empty_data(temp_outdir):
    """Test behavior with empty input data."""
    # Create empty DataFrame
    empty_df = pd.DataFrame()
    empty_path = temp_outdir / "normalized.parquet"
    pq.write_table(pa.Table.from_pandas(empty_df), empty_path)
    
    ctx = {
        "outdir": str(temp_outdir),
        "logger": None
    }
    
    result = google_maps_search.run({}, ctx)
    
    assert result['status'] == 'SKIPPED'
    assert 'EMPTY_INPUT' in result['reason']