"""Tests to verify the fix for empty dataset.csv issue with missing contact information."""

import pandas as pd
import pytest
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from normalize.standardize import run as run_standardize
from package.exporter import run as run_exporter


def test_case_insensitive_column_matching():
    """Test that column matching works with different case variations."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test data with mixed case columns (common issue in real SIRENE data)
        input_data = {
            'SIREN': ['123456789', '987654321'],
            'SIRET': ['12345678900001', '98765432100001'],
            'DENOMINATIONUNITELEGALE': ['Expert Comptable SA', 'Cabinet Conseil SARL'],  # lowercase
            'LIBELLECOMMUNEETABLISSEMENT': ['Paris', 'Lyon'],  # uppercase
            'codePostalEtablissement': ['75008', '69001'],  # mixed case
            'adresseEtablissement': ['123 Avenue des Champs-Élysées', '456 Rue de la République'],
            'ACTIVITEPRINCIPALEUNITELLEGALE': ['6920Z', '6920Z'],  # uppercase, different column name
            'TELEPHONE': ['+33142123456', '0472345678'],  # uppercase
            'EMAIL': ['contact@expert.fr', 'info@cabinet.com'],  # uppercase
            'nomunitelegale': ['Dupont', 'Martin'],  # lowercase
            'prenomsunitelegale': ['Jean', 'Marie'],  # lowercase
            'etatAdministratifEtablissement': ['A', 'A']
        }
        
        input_df = pd.DataFrame(input_data)
        input_path = tmpdir / "input.parquet"
        pq.write_table(pa.Table.from_pandas(input_df), input_path)
        
        # Run standardization
        ctx = {
            'input_path': str(input_path),
            'outdir_path': tmpdir,
            'outdir': str(tmpdir),
            'job': {
                'filters': {
                    'naf_include': ['6920Z']
                }
            }
        }
        
        result = run_standardize({}, ctx)
        
        # Check that standardization succeeded
        assert result['status'] == 'OK'
        assert result['rows'] == 2
        
        # Read the normalized output
        normalized_path = tmpdir / "normalized.parquet"
        assert normalized_path.exists()
        
        normalized_df = pd.read_parquet(normalized_path)
        
        # Verify that contact information was properly extracted despite case differences
        assert normalized_df['telephone_norm'].notna().sum() == 2, "Phone numbers should be extracted"
        assert normalized_df['email'].notna().sum() == 2, "Emails should be extracted"
        assert normalized_df['nom'].notna().sum() == 2, "Names should be extracted"
        assert normalized_df['prenom'].notna().sum() == 2, "First names should be extracted"
        assert normalized_df['adresse'].notna().sum() == 2, "Addresses should be extracted"


def test_placeholder_value_filtering():
    """Test that placeholder values are properly filtered out."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test data with placeholder values (common issue in real data)
        input_data = {
            'siren': ['123456789', '987654321', '555555555'],
            'siret': ['12345678900001', '98765432100001', '55555555500001'],
            'denominationUniteLegale': ['Expert Comptable SA', 'DENOMINATION NON RENSEIGNEE', 'Cabinet OK'],
            'libelleCommuneEtablissement': ['Paris', 'Lyon', 'Marseille'],
            'codePostalEtablissement': ['75008', '69001', '13001'],
            'adresseEtablissement': ['123 Avenue des Champs-Élysées', 'ADRESSE NON RENSEIGNEE', '789 Boulevard Canebière'],
            'activitePrincipaleEtablissement': ['6920Z', '6920Z', '6920Z'],
            'telephone': ['+33142123456', 'TELEPHONE NON RENSEIGNE', '0491123456'],
            'email': ['contact@expert.fr', '', 'hello@cabinet.fr'],
            'nomUniteLegale': ['Dupont', 'Martin', 'Bernard'],
            'prenomsUniteLegale': ['Jean', 'Marie', 'Pierre'],
            'etatAdministratifEtablissement': ['A', 'A', 'A']
        }
        
        input_df = pd.DataFrame(input_data)
        input_path = tmpdir / "input.parquet"
        pq.write_table(pa.Table.from_pandas(input_df), input_path)
        
        # Run standardization
        ctx = {
            'input_path': str(input_path),
            'outdir_path': tmpdir,
            'outdir': str(tmpdir),
            'job': {
                'filters': {
                    'naf_include': ['6920Z']
                }
            }
        }
        
        result = run_standardize({}, ctx)
        
        # Check that standardization succeeded
        assert result['status'] == 'OK'
        assert result['rows'] == 3
        
        # Read the normalized output
        normalized_path = tmpdir / "normalized.parquet"
        normalized_df = pd.read_parquet(normalized_path)
        
        # Verify that placeholder values were filtered out
        assert normalized_df['telephone_norm'].notna().sum() == 2, "Placeholder phone should be filtered out"
        assert normalized_df['adresse'].notna().sum() == 2, "Placeholder address should be filtered out" 
        assert normalized_df['raison_sociale'].notna().sum() == 2, "Placeholder denomination should be filtered out"
        
        # Verify that valid data is preserved
        valid_phones = normalized_df['telephone_norm'].dropna().tolist()
        assert '33142123456' in valid_phones, "Valid phone should be preserved and normalized"
        assert '+33491123456' in valid_phones, "Valid phone should be preserved and normalized"
        
        valid_addresses = normalized_df['adresse'].dropna().tolist()
        assert '123 Avenue des Champs-Élysées' in valid_addresses, "Valid address should be preserved"
        assert '789 Boulevard Canebière' in valid_addresses, "Valid address should be preserved"


def test_complete_pipeline_produces_non_empty_dataset():
    """Test that the complete pipeline produces a non-empty dataset.csv with contact information."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create realistic test data that should produce a non-empty result
        input_data = {
            'siren': ['123456789', '987654321', '555555555'],
            'siret': ['12345678900001', '98765432100001', '55555555500001'],
            'denominationUniteLegale': ['Expert Comptable SA', 'Cabinet Conseil SARL', 'Société Expertise'],
            'libelleCommuneEtablissement': ['Paris', 'Lyon', 'Marseille'],
            'codePostalEtablissement': ['75008', '69001', '13001'],
            'adresseEtablissement': ['123 Avenue des Champs-Élysées', '456 Rue de la République', '789 Boulevard Canebière'],
            'activitePrincipaleEtablissement': ['6920Z', '6920Z', '6920Z'],  # All accounting firms
            'trancheEffectifsEtablissement': ['10-19', '20-49', '5-9'],
            'telephone': ['+33142123456', '0472345678', '04.91.12.34.56'],
            'email': ['contact@expert.fr', 'info@cabinet.com', 'hello@expertise.fr'],
            'siteweb': ['https://www.expert.fr', 'https://cabinet.com', ''],
            'nomUniteLegale': ['Dupont', 'Martin', 'Bernard'],
            'prenomsUniteLegale': ['Jean', 'Marie', 'Pierre'],
            'etatAdministratifEtablissement': ['A', 'A', 'A']  # All active
        }
        
        input_df = pd.DataFrame(input_data)
        input_path = tmpdir / "input.parquet"
        pq.write_table(pa.Table.from_pandas(input_df), input_path)
        
        # Step 1: Standardization
        standardize_ctx = {
            'input_path': str(input_path),
            'outdir_path': tmpdir,
            'outdir': str(tmpdir),
            'job': {
                'filters': {
                    'naf_include': ['6920Z']
                }
            }
        }
        
        result1 = run_standardize({}, standardize_ctx)
        assert result1['status'] == 'OK'
        assert result1['rows'] == 3
        
        # Step 2: Export to dataset.csv
        export_ctx = {
            'outdir_path': tmpdir,
            'outdir': str(tmpdir),
            'run_id': 'test_run_complete',
            'lang': 'fr'
        }
        
        result2 = run_exporter({}, export_ctx)
        assert result2['status'] == 'OK'
        
        # Verify that dataset.csv was created and is not empty
        dataset_csv = tmpdir / "dataset.csv"
        assert dataset_csv.exists(), "dataset.csv should be created"
        
        dataset_size = dataset_csv.stat().st_size
        assert dataset_size > 0, "dataset.csv should not be empty"
        
        # Read and verify the dataset content
        dataset_df = pd.read_csv(dataset_csv)
        assert len(dataset_df) == 3, "dataset.csv should contain all 3 companies"
        
        # Verify that contact information is present
        contact_fields = ['telephone_norm', 'nom', 'prenom', 'adresse', 'email']
        for field in contact_fields:
            assert field in dataset_df.columns, f"Contact field '{field}' should be in dataset"
            non_null_count = dataset_df[field].notna().sum()
            assert non_null_count > 0, f"Contact field '{field}' should have some non-null values"
        
        # Verify specific contact information is preserved
        assert dataset_df['telephone_norm'].notna().sum() == 3, "All phone numbers should be extracted"
        assert dataset_df['nom'].notna().sum() == 3, "All names should be extracted"
        assert dataset_df['prenom'].notna().sum() == 3, "All first names should be extracted"
        assert dataset_df['adresse'].notna().sum() == 3, "All addresses should be extracted"
        assert dataset_df['email'].notna().sum() == 3, "All emails should be extracted"
        
        # Verify data quality - phone numbers should be normalized
        phone_values = dataset_df['telephone_norm'].dropna().tolist()
        assert all(str(phone).startswith(('+33', '33')) for phone in phone_values), \
            "Phone numbers should be normalized to French format"


def test_naf_code_filtering_works():
    """Test that NAF code filtering works correctly and doesn't result in empty output."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test data with mixed NAF codes
        input_data = {
            'siren': ['123456789', '987654321', '555555555', '444444444'],
            'siret': ['12345678900001', '98765432100001', '55555555500001', '44444444400001'],
            'denominationUniteLegale': ['Expert Comptable SA', 'Boulangerie SARL', 'Cabinet Conseil', 'Restaurant XYZ'],
            'activitePrincipaleEtablissement': ['6920Z', '1071C', '6920Z', '5610A'],  # Only 2 accounting firms
            'telephone': ['+33142123456', '0143567890', '0472345678', '0156789012'],
            'etatAdministratifEtablissement': ['A', 'A', 'A', 'A']
        }
        
        input_df = pd.DataFrame(input_data)
        input_path = tmpdir / "input.parquet"
        pq.write_table(pa.Table.from_pandas(input_df), input_path)
        
        # Run standardization with NAF filter for accounting firms only
        ctx = {
            'input_path': str(input_path),
            'outdir_path': tmpdir,
            'outdir': str(tmpdir),
            'job': {
                'filters': {
                    'naf_include': ['6920Z']  # Only accounting firms
                }
            }
        }
        
        result = run_standardize({}, ctx)
        
        assert result['status'] == 'OK'
        assert result['rows'] == 2, "Should only process companies with NAF code 6920Z"
        
        # Verify the right companies were selected
        normalized_path = tmpdir / "normalized.parquet"
        normalized_df = pd.read_parquet(normalized_path)
        
        company_names = normalized_df['raison_sociale'].tolist()
        assert 'Expert Comptable SA' in company_names, "Accounting firm should be included"
        assert 'Cabinet Conseil' in company_names, "Accounting firm should be included"
        assert 'Boulangerie SARL' not in company_names, "Bakery should be filtered out"
        assert 'Restaurant XYZ' not in company_names, "Restaurant should be filtered out"