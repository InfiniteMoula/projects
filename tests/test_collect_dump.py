"""Tests for dumps.collect_dump module to ensure it doesn't waste disk space."""

import tempfile
from pathlib import Path

import pytest

from dumps import collect_dump


def test_collect_dump_copies_files():
    """Test that collect_dump properly copies files to the dumps directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a test input file
        input_file = tmp_path / "input.parquet"
        test_content = b"test data content that should be copied"
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
        
        # Check that it successfully copied the file
        assert result["status"] == "OK"
        assert result["source_file"] == str(input_file)
        assert result["file_size_bytes"] == len(test_content)
        assert "destination_file" in result
        
        # Verify that file was actually copied to dumps folder
        dumps_dir = outdir / "dumps"
        assert dumps_dir.exists()
        copied_files = list(dumps_dir.iterdir())
        assert len(copied_files) == 1
        assert copied_files[0].name == "input.parquet"
        assert copied_files[0].read_bytes() == test_content
        

def test_collect_dump_downloads_http_sources():
    """Test that collect_dump downloads HTTP sources to the dumps directory."""
    import httpx
    from unittest.mock import patch, MagicMock
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        outdir = tmp_path / "output"
        
        ctx = {
            "outdir": str(outdir),
            "outdir_path": outdir,
            "input": "https://example.com/test-file.csv"
        }
        
        # Mock the stream_download function to simulate successful download
        test_content = b"CSV header\ntest,data,content"
        with patch('dumps.collect_dump.stream_download') as mock_download:
            mock_path = outdir / "dumps" / "test-file.csv"
            mock_path.parent.mkdir(parents=True, exist_ok=True)
            mock_path.write_bytes(test_content)
            mock_download.return_value = mock_path
            
            result = collect_dump.run({}, ctx)
            
            # Check that it successfully downloaded the file
            assert result["status"] == "OK"
            assert result["source_url"] == "https://example.com/test-file.csv"
            assert result["file_size_bytes"] == len(test_content)
            assert "destination_file" in result
            
            # Verify the download function was called correctly
            mock_download.assert_called_once()
            args, kwargs = mock_download.call_args
            assert args[0] == "https://example.com/test-file.csv"
            assert str(args[1]).endswith("test-file.csv")


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


def test_collect_dump_handles_http_download_failure():
    """Test that collect_dump handles HTTP download failures gracefully."""
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        outdir = tmp_path / "output"
        
        ctx = {
            "outdir": str(outdir),
            "outdir_path": outdir,
            "input": "https://example.com/nonexistent-file.csv"
        }
        
        # Mock stream_download to raise an exception
        with patch('dumps.collect_dump.stream_download') as mock_download:
            mock_download.side_effect = Exception("Download failed")
            
            result = collect_dump.run({}, ctx)
            
            # Check that it returns an error status
            assert result["status"] == "ERROR"
            assert "Download failed" in result["error"]
            assert result["source_url"] == "https://example.com/nonexistent-file.csv"


def test_collect_dump_handles_file_copy_failure():
    """Test that collect_dump handles file copy failures gracefully."""
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a test input file
        input_file = tmp_path / "input.parquet"
        test_content = b"test data content"
        input_file.write_bytes(test_content)
        
        # Setup context
        outdir = tmp_path / "output"
        ctx = {
            "outdir": str(outdir),
            "outdir_path": outdir,
            "input": str(input_file)
        }
        
        # Mock shutil.copy2 to raise an exception
        with patch('shutil.copy2') as mock_copy:
            mock_copy.side_effect = Exception("Copy failed")
            
            result = collect_dump.run({}, ctx)
            
            # Check that it returns an error status
            assert result["status"] == "ERROR"
            assert "Copy failed" in result["error"]
            assert result["source_file"] == str(input_file)