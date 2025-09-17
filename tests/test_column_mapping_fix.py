"""Tests to verify the column mapping fix works correctly."""

import pandas as pd
import pytest
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from normalize.standardize import run as run_standardize
from quality.score import run as run_quality_score


def test_column_mapping_fix():
    """Test that the standardization creates both original and aliased column names."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test input data with SIRENE format
        input_data = {
            'siren': ['123456789', '987654321'],
            'siret': ['12345678900001', '98765432100001'],
            'denominationUniteLegale': ['Entreprise Test SA', 'Société Exemple SARL'],
            'libelleCommuneEtablissement': ['Paris', 'Lyon'],
            'codePostalEtablissement': ['75001', '69001'],
            'adresseEtablissement': ['123 Rue de la Paix', '456 Avenue des Fleurs'],
            'activitePrincipaleEtablissement': ['6202A', '6920Z'],
            'trancheEffectifsEtablissement': ['20-49', '10-19'],
            'telephone': ['+33142123456', '0472345678'],
            'email': ['contact@test.fr', 'info@exemple.fr'],
            'etatAdministratifEtablissement': ['A', 'A']
        }
        
        input_df = pd.DataFrame(input_data)
        input_path = tmpdir / "input.parquet"
        pq.write_table(pa.Table.from_pandas(input_df), input_path)
        
        # Run standardization
        cfg = {}
        ctx = {
            'input_path': str(input_path),
            'outdir_path': tmpdir,
            'outdir': str(tmpdir)
        }
        
        result = run_standardize(cfg, ctx)
        
        # Check that standardization succeeded
        assert result['status'] == 'OK'
        assert result['rows'] == 2
        
        # Read the normalized output
        normalized_path = tmpdir / "normalized.parquet"
        assert normalized_path.exists()
        
        normalized_df = pd.read_parquet(normalized_path)
        
        # Verify original columns exist
        original_columns = ['siren', 'siret', 'raison_sociale', 'commune', 'cp', 'adresse', 'naf']
        for col in original_columns:
            assert col in normalized_df.columns, f"Missing original column: {col}"
        
        # Verify aliased columns exist  
        aliased_columns = ['denomination', 'ville', 'code_postal', 'adresse_complete', 'naf_code']
        for col in aliased_columns:
            assert col in normalized_df.columns, f"Missing aliased column: {col}"
        
        # Verify new effectif column exists
        assert 'effectif' in normalized_df.columns, "Missing effectif column"
        
        # Verify content consistency between original and aliased columns
        assert (normalized_df['raison_sociale'] == normalized_df['denomination']).all()
        assert (normalized_df['commune'] == normalized_df['ville']).all()
        assert (normalized_df['cp'] == normalized_df['code_postal']).all()
        assert (normalized_df['adresse'] == normalized_df['adresse_complete']).all()
        assert (normalized_df['naf'] == normalized_df['naf_code']).all()
        
        # Verify effectif data is extracted
        assert not normalized_df['effectif'].isna().all(), "Effectif should contain some data"


def test_quality_scoring_with_column_mapping():
    """Test that quality scoring now works correctly with the mapped columns."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test input data with SIRENE format
        input_data = {
            'siren': ['123456789'],
            'siret': ['12345678900001'],
            'denominationUniteLegale': ['Excellent Company SA'],
            'libelleCommuneEtablissement': ['Paris'],
            'codePostalEtablissement': ['75008'],
            'adresseEtablissement': ['123 Avenue des Champs-Élysées'],
            'activitePrincipaleEtablissement': ['6202A'],
            'trancheEffectifsEtablissement': ['50-99'],
            'telephone': ['+33142123456'],
            'email': ['contact@excellent.fr'],
            'etatAdministratifEtablissement': ['A']
        }
        
        input_df = pd.DataFrame(input_data)
        input_path = tmpdir / "input.parquet"
        pq.write_table(pa.Table.from_pandas(input_df), input_path)
        
        # Run standardization
        ctx = {
            'input_path': str(input_path),
            'outdir_path': tmpdir,
            'outdir': str(tmpdir)
        }
        
        standardize_result = run_standardize({}, ctx)
        assert standardize_result['status'] == 'OK'
        
        # Run quality scoring
        cfg = {'scoring': {'weights': {'contactability': 50, 'unicity': 20, 'completeness': 20, 'freshness': 10}}}
        
        score_result = run_quality_score(cfg, ctx)
        
        # Check that scoring succeeded - if it fails, print the error for debugging
        if score_result['status'] != 'OK':
            print(f"Quality scoring failed: {score_result}")
            # List files in temp directory to debug
            print(f"Files in {tmpdir}: {list(tmpdir.iterdir())}")
        
        assert score_result['status'] == 'OK'
        assert score_result['rows'] == 1
        
        # The key improvement: completeness should be significantly higher now
        # because the expected columns (denomination, naf_code, etc.) exist
        assert score_result['score_mean'] > 0.5, f"Quality score should be higher than 0.5, got {score_result['score_mean']}"
        
        # Read the quality scores
        quality_path = tmpdir / "quality_score.parquet"
        assert quality_path.exists()
        
        quality_df = pd.read_parquet(quality_path)
        assert 'score_quality' in quality_df.columns
        assert not quality_df['score_quality'].isna().all()


def test_expected_columns_in_quality_calculation():
    """Test that the quality calculation can find all expected columns."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test data that should have high completeness
        input_data = {
            'siren': ['123456789'],
            'siret': ['12345678900001'],
            'denominationUniteLegale': ['Test Company'],
            'libelleCommuneEtablissement': ['Paris'],
            'codePostalEtablissement': ['75001'],
            'adresseEtablissement': ['123 Test Street'],
            'activitePrincipaleEtablissement': ['6202A'],
            'trancheEffectifsEtablissement': ['10-19'],
            'etatAdministratifEtablissement': ['A']
        }
        
        input_df = pd.DataFrame(input_data)
        input_path = tmpdir / "input.parquet"
        pq.write_table(pa.Table.from_pandas(input_df), input_path)
        
        # Run standardization
        ctx = {
            'input_path': str(input_path),
            'outdir_path': tmpdir,
            'outdir': str(tmpdir)
        }
        
        standardize_result = run_standardize({}, ctx)
        assert standardize_result['status'] == 'OK'
        
        # Load the normalized data and check all expected quality fields
        normalized_df = pd.read_parquet(tmpdir / "normalized.parquet")
        
        # These are the fields that quality/score.py checks for completeness
        expected_quality_fields = [
            "siren", "denomination", "raison_sociale", "naf_code", 
            "adresse_complete", "code_postal", "ville", "effectif"
        ]
        
        for field in expected_quality_fields:
            assert field in normalized_df.columns, f"Expected quality field {field} missing from normalized data"
            
        # Verify that most fields have actual data (not all null)
        non_null_counts = {field: normalized_df[field].notna().sum() for field in expected_quality_fields}
        
        # At least these core fields should have data
        core_fields = ["siren", "denomination", "naf_code", "code_postal", "ville"]
        for field in core_fields:
            assert non_null_counts[field] > 0, f"Core field {field} should have non-null data"