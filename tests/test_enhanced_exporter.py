"""Tests for the enhanced package exporter."""

import json
import pytest
import tempfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

import package.exporter as exporter


@pytest.fixture
def temp_outdir():
    """Create a temporary output directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        
        # Create sample deduped.parquet
        deduped_data = {
            'siren': ['123456789', '987654321'],
            'raison_sociale': ['Test Company A', 'Test Company B'],
            'domain_root': ['test-a.fr', 'test-b.com'],
            'best_email': ['contact@test-a.fr', 'info@test-b.com'],
            'contactability': [0.85, 0.92],
            'unicity': [0.95, 0.88],
            'completeness': [0.90, 0.85],
            'freshness': [0.80, 0.95]
        }
        deduped_df = pd.DataFrame(deduped_data)
        pq.write_table(pa.Table.from_pandas(deduped_df), outdir / "deduped.parquet")
        
        # Create sample quality_score.parquet
        quality_data = {'score_quality': [0.875, 0.905]}
        quality_df = pd.DataFrame(quality_data)
        pq.write_table(pa.Table.from_pandas(quality_df), outdir / "quality_score.parquet")
        
        yield outdir


def test_merge_quality_data(temp_outdir):
    """Test merging deduped data with quality scores."""
    
    df = exporter.merge_quality_data(temp_outdir)
    
    assert len(df) == 2
    assert 'score_quality' in df.columns
    assert 'siren' in df.columns
    assert df['score_quality'].iloc[0] == 0.875


def test_calculate_data_dictionary(temp_outdir):
    """Test data dictionary calculation."""
    
    df = exporter.merge_quality_data(temp_outdir)
    data_dict = exporter.calculate_data_dictionary(df)
    
    assert len(data_dict) > 0
    assert all('column' in item for item in data_dict)
    assert all('non_null' in item for item in data_dict)
    assert all('completeness_rate' in item for item in data_dict)
    
    # Check completeness calculation
    siren_entry = next(item for item in data_dict if item['column'] == 'siren')
    assert siren_entry['non_null'] == 2
    assert siren_entry['completeness_rate'] == 100.0


def test_calculate_quality_metrics(temp_outdir):
    """Test quality metrics calculation."""
    
    df = exporter.merge_quality_data(temp_outdir)
    metrics = exporter.calculate_quality_metrics(df)
    
    assert metrics['total_records'] == 2
    assert metrics['quality_mean'] is not None
    assert metrics['quality_p50'] is not None
    assert metrics['quality_p90'] is not None
    
    # Quality scores should be in percentage (0-100)
    assert 0 <= metrics['quality_mean'] <= 100


def test_enhanced_exporter_run(temp_outdir):
    """Test the complete enhanced exporter run."""
    
    ctx = {
        "run_id": "test_run_456",
        "outdir": str(temp_outdir),
        "outdir_path": temp_outdir,
        "job_path": None,
        "lang": "fr"
    }
    
    cfg = {}
    
    result = exporter.run(cfg, ctx)
    
    # Check result status
    assert result['status'] == 'OK'
    assert 'csv' in result
    assert 'parquet' in result
    assert 'html_report' in result
    assert 'total_records' in result
    
    # Check that files were created
    assert (temp_outdir / "dataset.csv").exists()
    assert (temp_outdir / "dataset.parquet").exists()
    assert (temp_outdir / "data_quality_report.html").exists()
    assert (temp_outdir / "manifest.json").exists()
    assert (temp_outdir / "data_dictionary.md").exists()
    assert (temp_outdir / "sha256.txt").exists()
    
    # Check CSV content
    csv_df = pd.read_csv(temp_outdir / "dataset.csv")
    assert len(csv_df) == 2
    assert 'score_quality' in csv_df.columns
    
    # Check manifest content
    with open(temp_outdir / "manifest.json") as f:
        manifest = json.load(f)
    
    assert manifest['run_id'] == 'test_run_456'
    assert manifest['records'] == 2
    assert 'quality_metrics' in manifest
    assert 'paths' in manifest
    assert manifest['paths']['html_report'] is not None


def test_exporter_with_missing_quality_data(temp_outdir):
    """Test exporter when quality_score.parquet is missing."""
    
    # Remove quality scores file
    (temp_outdir / "quality_score.parquet").unlink()
    
    ctx = {
        "run_id": "test_run_789",
        "outdir": str(temp_outdir),
        "outdir_path": temp_outdir,
        "job_path": None,
        "lang": "fr"
    }
    
    result = exporter.run({}, ctx)
    
    # Should still work without quality scores
    assert result['status'] == 'OK'
    assert (temp_outdir / "dataset.csv").exists()
    
    # Quality metrics should be None
    assert result['quality_mean'] is None


def test_exporter_with_no_data_files(temp_outdir):
    """Test exporter behavior when no data files are present."""
    
    # Remove all parquet files
    for file in temp_outdir.glob("*.parquet"):
        file.unlink()
    
    ctx = {
        "run_id": "test_run_error",
        "outdir": str(temp_outdir),
        "outdir_path": temp_outdir,
        "job_path": None,
        "lang": "fr"
    }
    
    result = exporter.run({}, ctx)
    
    # Should fail gracefully
    assert result['status'] == 'FAIL'
    assert 'error' in result