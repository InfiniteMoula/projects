"""Tests for the quality scoring algorithm fix."""

import pytest
import tempfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

import quality.score as quality_score


@pytest.fixture
def temp_outdir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_quality_scoring_with_complete_data(temp_outdir):
    """Test quality scoring with complete, high-quality data."""
    
    # Create high-quality sample data
    data = {
        'siren': ['123456789'],
        'raison_sociale': ['Excellent Company SA'],
        'denomination': ['Excellent Company'],
        'naf_code': ['6202A'],
        'adresse_complete': ['123 Avenue des Champs-Élysées, 75008 Paris'],
        'code_postal': ['75008'],
        'ville': ['Paris'],
        'domain_root': ['excellent-company.fr'],
        'best_email': ['contact@excellent-company.fr'],
        'telephone_norm': ['+33142123456'],
        'effectif': [150],
        'email_plausible': [0.95],
        'domain_valid': [1.0]
    }
    df = pd.DataFrame(data)
    pq.write_table(pa.Table.from_pandas(df), temp_outdir / "deduped.parquet")
    
    # Run scoring
    cfg = {'scoring': {'weights': {'contactability': 50, 'unicity': 20, 'completeness': 20, 'freshness': 10}}}
    ctx = {'outdir_path': temp_outdir}
    
    result = quality_score.run(cfg, ctx)
    
    # Check results
    assert result['status'] == 'OK'
    assert result['rows'] == 1
    assert result['score_mean'] > 0.9  # Should be high quality
    
    # Check that quality score file was created
    quality_df = pd.read_parquet(temp_outdir / "quality_score.parquet")
    assert len(quality_df) == 1
    assert quality_df['score_quality'].iloc[0] > 0.9


def test_quality_scoring_with_incomplete_data(temp_outdir):
    """Test quality scoring with incomplete, low-quality data."""
    
    # Create low-quality sample data
    data = {
        'siren': ['111111111'],
        'raison_sociale': [''],  # Missing
        'denomination': [''],    # Missing
        'naf_code': [''],        # Missing
        'adresse_complete': [''], # Missing
        'code_postal': ['13001'],
        'ville': ['Marseille'],
        'domain_root': [''],     # Missing
        'best_email': [''],      # Missing
        'telephone_norm': [''],  # Missing
        'effectif': ['']         # Missing
    }
    df = pd.DataFrame(data)
    pq.write_table(pa.Table.from_pandas(df), temp_outdir / "deduped.parquet")
    
    # Run scoring
    cfg = {'scoring': {'weights': {'contactability': 50, 'unicity': 20, 'completeness': 20, 'freshness': 10}}}
    ctx = {'outdir_path': temp_outdir}
    
    result = quality_score.run(cfg, ctx)
    
    # Check results
    assert result['status'] == 'OK'
    assert result['rows'] == 1
    assert result['score_mean'] < 0.5  # Should be low quality
    
    # Check that quality score file was created
    quality_df = pd.read_parquet(temp_outdir / "quality_score.parquet")
    assert len(quality_df) == 1
    assert quality_df['score_quality'].iloc[0] < 0.5


def test_quality_scoring_with_mixed_data(temp_outdir):
    """Test quality scoring with mixed quality data."""
    
    # Create data with varying quality levels
    data = {
        'siren': ['123456789', '987654321', '111111111'],
        'raison_sociale': ['High Quality Company', 'Medium Company', ''],
        'denomination': ['HQ Company Ltd', 'Medium Co', 'Low Co'],
        'naf_code': ['6202A', '7112B', ''],
        'adresse_complete': ['123 Complete Address Paris', '456 Short St', ''],
        'code_postal': ['75001', '69001', '13001'],
        'ville': ['Paris', 'Lyon', 'Marseille'],
        'domain_root': ['hq-company.fr', 'medium.com', ''],
        'best_email': ['contact@hq-company.fr', 'info@medium.com', ''],
        'telephone_norm': ['+33123456789', '', ''],
        'effectif': [100, 50, 10]
    }
    df = pd.DataFrame(data)
    pq.write_table(pa.Table.from_pandas(df), temp_outdir / "deduped.parquet")
    
    # Run scoring
    cfg = {'scoring': {'weights': {'contactability': 50, 'unicity': 20, 'completeness': 20, 'freshness': 10}}}
    ctx = {'outdir_path': temp_outdir}
    
    result = quality_score.run(cfg, ctx)
    
    # Check results
    assert result['status'] == 'OK'
    assert result['rows'] == 3
    
    # Quality scores should not be uniform
    quality_df = pd.read_parquet(temp_outdir / "quality_score.parquet")
    scores = quality_df['score_quality'].tolist()
    
    # Should have different scores (not all the same)
    assert len(set([round(s, 2) for s in scores])) > 1, f"Scores should not be uniform: {scores}"
    
    # First record should have highest score (most complete)
    assert scores[0] > scores[1], f"First record should score higher: {scores}"
    assert scores[1] > scores[2], f"Second record should score higher than third: {scores}"


def test_quality_scoring_with_duplicates(temp_outdir):
    """Test that duplicate detection affects unicity scoring."""
    
    # Create data with duplicates
    data = {
        'siren': ['123456789', '123456789', '987654321'],  # Duplicate SIREN
        'raison_sociale': ['Company A', 'Company A', 'Company B'],
        'domain_root': ['company-a.fr', 'company-a.fr', 'company-b.fr'],
        'best_email': ['contact@company-a.fr', 'contact@company-a.fr', 'info@company-b.fr'],
        'telephone_norm': ['+33123456789', '+33123456789', '+33987654321']
    }
    df = pd.DataFrame(data)
    pq.write_table(pa.Table.from_pandas(df), temp_outdir / "deduped.parquet")
    
    # Run scoring
    cfg = {'scoring': {'weights': {'contactability': 50, 'unicity': 20, 'completeness': 20, 'freshness': 10}}}
    ctx = {'outdir_path': temp_outdir}
    
    result = quality_score.run(cfg, ctx)
    
    # Check results
    assert result['status'] == 'OK'
    assert result['rows'] == 3
    
    quality_df = pd.read_parquet(temp_outdir / "quality_score.parquet")
    scores = quality_df['score_quality'].tolist()
    
    # The unique record (third one) should have a higher score than the duplicates
    assert scores[2] > scores[0], f"Unique record should score higher: {scores}"
    assert scores[2] > scores[1], f"Unique record should score higher: {scores}"


def test_quality_scoring_no_uniform_zeros(temp_outdir):
    """Test that the fixed algorithm doesn't produce uniform 0.0 scores."""
    
    # Create basic data without explicit quality metrics
    data = {
        'siren': ['123456789', '987654321'],
        'raison_sociale': ['Test Company A', 'Test Company B'],
        'domain_root': ['test-a.fr', 'test-b.com'],
        'best_email': ['contact@test-a.fr', 'info@test-b.com'],
        'telephone_norm': ['+33123456789', '']  # One complete, one incomplete
    }
    df = pd.DataFrame(data)
    pq.write_table(pa.Table.from_pandas(df), temp_outdir / "deduped.parquet")
    
    # Run scoring
    cfg = {'scoring': {'weights': {'contactability': 50, 'unicity': 20, 'completeness': 20, 'freshness': 10}}}
    ctx = {'outdir_path': temp_outdir}
    
    result = quality_score.run(cfg, ctx)
    
    # Check results
    assert result['status'] == 'OK'
    assert result['score_mean'] > 0.0, "Mean score should not be 0.0"
    
    quality_df = pd.read_parquet(temp_outdir / "quality_score.parquet")
    scores = quality_df['score_quality'].tolist()
    
    # Scores should not all be 0.0
    assert any(score > 0.0 for score in scores), f"At least some scores should be > 0.0: {scores}"
    assert not all(score == 0.0 for score in scores), f"Not all scores should be 0.0: {scores}"


def test_quality_scoring_custom_weights(temp_outdir):
    """Test quality scoring with custom weights."""
    
    # Create test data
    data = {
        'siren': ['123456789'],
        'raison_sociale': ['Test Company'],
        'best_email': ['contact@test.fr'],
        'telephone_norm': ['+33123456789'],
        'domain_root': ['test.fr']
    }
    df = pd.DataFrame(data)
    pq.write_table(pa.Table.from_pandas(df), temp_outdir / "deduped.parquet")
    
    # Test with contactability heavily weighted
    cfg = {'scoring': {'weights': {'contactability': 90, 'unicity': 5, 'completeness': 3, 'freshness': 2}}}
    ctx = {'outdir_path': temp_outdir}
    
    result = quality_score.run(cfg, ctx)
    
    assert result['status'] == 'OK'
    assert result['score_mean'] > 0.0
    
    # Score should be influenced by high contactability weight
    quality_df = pd.read_parquet(temp_outdir / "quality_score.parquet")
    score_with_contact_weight = quality_df['score_quality'].iloc[0]
    
    # Test with completeness heavily weighted
    cfg = {'scoring': {'weights': {'contactability': 10, 'unicity': 10, 'completeness': 70, 'freshness': 10}}}
    result2 = quality_score.run(cfg, ctx)
    
    # The results should be different (different weighting should affect final score)
    quality_df2 = pd.read_parquet(temp_outdir / "quality_score.parquet")
    score_with_completeness_weight = quality_df2['score_quality'].iloc[0]
    
    # Scores should be different with different weights
    assert abs(score_with_contact_weight - score_with_completeness_weight) > 0.01, \
        f"Different weights should produce different scores: {score_with_contact_weight} vs {score_with_completeness_weight}"