"""Test comprehensive extraction functionality."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq

from normalize.standardize import run


def test_comprehensive_extraction_captures_all_data():
    """Test that the new comprehensive extraction captures significantly more data than before."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test data with various column naming conventions that should all be captured
        input_data = {
            # Core SIRENE data
            'siren': ['123456789', '987654321'],
            'siret': ['12345678900001', '98765432100001'],
            'denominationUniteLegale': ['Expert Comptable SA', 'Société Exemple SARL'],
            'activitePrincipaleEtablissement': ['6920Z', '6920Z'],
            'etatAdministratifEtablissement': ['A', 'A'],
            
            # Contact variations that should be merged
            'telephone': ['+33142123456', '0472345678'],
            'tel_mobile': ['0612345678', '0687654321'],
            'fax': ['0142123457', '0472345679'],
            'email': ['contact@test.fr', 'info@exemple.fr'],
            'mail': ['alt@test.fr', 'alt@exemple.fr'],
            'website': ['www.test.fr', 'www.exemple.fr'],
            'site_web': ['test.fr', 'exemple.fr'],
            
            # Address variations that should be merged
            'adresseEtablissement': ['123 Rue de la Paix', '456 Avenue des Fleurs'],
            'libelleCommuneEtablissement': ['Paris', 'Lyon'],
            'codePostalEtablissement': ['75001', '69001'],
            
            # Name variations that should be merged  
            'nom': ['MARTIN', 'DUPONT'],
            'prenom': ['Jean', 'Pierre'],
            'noms': ['MARTIN Jean', 'DUPONT Pierre'],
            
            # Additional business data that should be preserved
            'capital_social': ['50000', '10000'],
            'forme_juridique': ['SA', 'SARL'],
            'secteur_activite': ['Comptabilité', 'Conseil'],
            'trancheEffectifsEtablissement': ['20-49', '10-19'],
            'date_creation': ['2010-01-01', '2015-06-15'],
        }
        
        input_df = pd.DataFrame(input_data)
        input_path = tmpdir / "input.parquet"
        pq.write_table(pa.Table.from_pandas(input_df), input_path)
        
        # Run standardization with NAF filter
        ctx = {
            'input_path': str(input_path),
            'outdir_path': str(tmpdir),
            'outdir': str(tmpdir),
            'job': {
                'filters': {
                    'naf_include': ['6920Z']
                }
            }
        }
        
        result = run({}, ctx)
        
        # Verify the extraction succeeded
        assert result['status'] == 'OK'
        assert result['rows'] == 2
        
        # Read the output
        normalized_path = tmpdir / "normalized.parquet"
        normalized_df = pd.read_parquet(normalized_path)
        
        # Should have significantly more columns than the old fixed schema (20 columns)
        assert len(normalized_df.columns) > 20, f"Expected more than 20 columns, got {len(normalized_df.columns)}"
        
        # Verify phone number variations are captured
        phone_columns = [col for col in normalized_df.columns if 'telephone' in col or 'fax' in col]
        assert len(phone_columns) >= 3, f"Expected at least 3 phone-related columns, got {phone_columns}"
        
        # Verify multiple contact methods are captured
        contact_columns = [col for col in normalized_df.columns if col in ['email', 'website', 'siteweb']]
        assert len(contact_columns) >= 2, f"Expected multiple contact columns, got {contact_columns}"
        
        # Verify business information is captured
        business_columns = [col for col in normalized_df.columns if col in ['capital_social', 'forme_juridique', 'secteur_activite']]
        assert len(business_columns) >= 3, f"Expected business info columns, got {business_columns}"
        
        # Verify backward compatibility - legacy column names should still exist
        legacy_columns = ['siren', 'siret', 'raison_sociale', 'adresse', 'naf', 'telephone_norm']
        for col in legacy_columns:
            assert col in normalized_df.columns, f"Missing backward compatibility column: {col}"
        
        # Verify data quality - no rows should be lost
        assert len(normalized_df) == 2
        
        # Verify normalized phone numbers
        assert 'telephone_norm' in normalized_df.columns
        assert normalized_df['telephone_norm'].notna().all(), "Phone normalization should work"
        
        # Verify NAF filtering worked
        assert normalized_df['naf'].notna().all()
        assert all(normalized_df['naf'].str.contains('6920', na=False)), "NAF filtering should work"


def test_comprehensive_extraction_preserves_all_column_types():
    """Test that columns with unconventional names are still preserved."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create data with unusual column names that should be preserved
        input_data = {
            'siren': ['123456789'],
            'activitePrincipaleEtablissement': ['6920Z'],
            'etatAdministratifEtablissement': ['A'],
            
            # Unusual column names that should be preserved
            'custom_field_1': ['Value 1'],
            'SPECIAL_DATA': ['Special Value'],
            'metadata.info': ['Meta Info'],
            'field-with-dashes': ['Dash Value'],
            'field with spaces': ['Space Value'],
        }
        
        input_df = pd.DataFrame(input_data)
        input_path = tmpdir / "input.parquet"
        pq.write_table(pa.Table.from_pandas(input_df), input_path)
        
        ctx = {
            'input_path': str(input_path),
            'outdir_path': str(tmpdir),
            'outdir': str(tmpdir),
            'job': {
                'filters': {
                    'naf_include': ['6920Z']
                }
            }
        }
        
        result = run({}, ctx)
        assert result['status'] == 'OK'
        
        normalized_df = pd.read_parquet(tmpdir / "normalized.parquet")
        
        # Check that unusual columns are preserved (possibly with cleaned names)
        output_columns = set(normalized_df.columns)
        
        # Should have some form of these custom fields
        custom_columns = [col for col in output_columns if 'custom' in col.lower() or 'special' in col.lower() or 'meta' in col.lower()]
        assert len(custom_columns) >= 3, f"Custom columns should be preserved, got {custom_columns}"


def test_comprehensive_extraction_merges_similar_columns():
    """Test that similar columns are properly merged."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create data with multiple similar columns that should be merged
        input_data = {
            'siren': ['123456789'],
            'activitePrincipaleEtablissement': ['6920Z'],
            'etatAdministratifEtablissement': ['A'],
            
            # Multiple phone columns - should be merged intelligently
            'telephone': ['0142123456'],
            'phone': ['0143567890'],  # Should go to telephone group
            'tel_mobile': ['0612345678'],  # Should go to telephone_mobile group
            'mobile': ['0687654321'],  # Should go to telephone_mobile group
            'fax': ['0142123457'],  # Should go to fax group
            
            # Multiple email columns - should be merged
            'email': ['primary@test.fr'],
            'mail': ['secondary@test.fr'],
            'e_mail': ['tertiary@test.fr'],
            
            # Multiple name columns - should be merged
            'nom': ['MARTIN'],
            'name': ['Jean MARTIN'],
            'noms': ['MARTIN Jean'],
        }
        
        input_df = pd.DataFrame(input_data)
        input_path = tmpdir / "input.parquet"
        pq.write_table(pa.Table.from_pandas(input_df), input_path)
        
        ctx = {
            'input_path': str(input_path),
            'outdir_path': str(tmpdir),
            'outdir': str(tmpdir),
            'job': {
                'filters': {
                    'naf_include': ['6920Z']
                }
            }
        }
        
        result = run({}, ctx)
        assert result['status'] == 'OK'
        
        normalized_df = pd.read_parquet(tmpdir / "normalized.parquet")
        
        # Should have separate telephone, telephone_mobile, and fax columns
        assert 'telephone' in normalized_df.columns
        assert 'telephone_mobile' in normalized_df.columns  
        assert 'fax' in normalized_df.columns
        
        # Should have normalized versions
        assert 'telephone_norm' in normalized_df.columns
        assert 'telephone_mobile_norm' in normalized_df.columns
        assert 'fax_norm' in normalized_df.columns
        
        # Email should be merged (picks first non-empty)
        assert 'email' in normalized_df.columns
        assert normalized_df['email'].iloc[0] == 'primary@test.fr'
        
        # Names should be merged
        assert 'nom' in normalized_df.columns
        assert normalized_df['nom'].notna().iloc[0]