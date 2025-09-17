"""Tests for dumps.collect_dump module to ensure it doesn't waste disk space."""

import tempfile
from pathlib import Path

import pytest

from dumps import collect_dump


def test_collect_dump_skips_file_copying():
    """Test that collect_dump no longer copies files to save disk space."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a test input file
        input_file = tmp_path / "input.parquet"
        test_content = b"test data content that would normally be copied"
        input_file.write_bytes(test_content)
        
        # Setup context
        outdir = tmp_path / "output"
        ctx = {
            "outdir": str(outdir),
            "outdir_path": outdir,
            "input": str(input_file)
        }
        
        # Run the collect_dump step
        result = collect_dump.run({}, ctx)
        
        # Check that it returns metadata instead of copying files
        assert result["status"] == "SKIPPED"
        assert result["reason"] == "RAW_DUMP_DISABLED"
        assert result["source_file"] == str(input_file)
        assert result["file_size_bytes"] == len(test_content)
        
        # Verify that no file was actually copied to dumps folder
        dumps_dir = outdir / "dumps"
        if dumps_dir.exists():
            # The directory might be created but should be empty
            assert len(list(dumps_dir.iterdir())) == 0
        

def test_collect_dump_skips_http_downloads():
    """Test that collect_dump no longer downloads HTTP sources."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        outdir = tmp_path / "output"
        
        ctx = {
            "outdir": str(outdir),
            "outdir_path": outdir,
            "input": "https://example.com/large-file.csv"
        }
        
        result = collect_dump.run({}, ctx)
        
        # Check that it returns metadata without downloading
        assert result["status"] == "SKIPPED"
        assert result["reason"] == "RAW_DUMP_DISABLED"
        assert result["source_url"] == "https://example.com/large-file.csv"
        
        # Verify no download occurred
        dumps_dir = outdir / "dumps"
        if dumps_dir.exists():
            assert len(list(dumps_dir.iterdir())) == 0


def test_collect_dump_handles_missing_input():
    """Test that collect_dump still handles missing input gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        outdir = tmp_path / "output"
        
        ctx = {
            "outdir": str(outdir),
            "outdir_path": outdir,
            "input": None
        }
        
        result = collect_dump.run({}, ctx)
        
        assert result["status"] == "SKIPPED"
        assert result["reason"] == "NO_INPUT"


def test_collect_dump_handles_nonexistent_file():
    """Test that collect_dump still raises error for nonexistent files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        outdir = tmp_path / "output"
        
        ctx = {
            "outdir": str(outdir),
            "outdir_path": outdir,
            "input": str(tmp_path / "nonexistent.parquet")
        }
        
        with pytest.raises(FileNotFoundError):
            collect_dump.run({}, ctx)