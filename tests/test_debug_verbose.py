import argparse
from pathlib import Path
import logging
import pytest
from unittest.mock import patch, MagicMock

import builder_cli
from utils import pipeline


def _namespace(tmp_path, **overrides):
    defaults = dict(
        job=str(tmp_path / "job.yml"),
        out=str(tmp_path / "out"),
        input=None,
        run_id=None,
        dry_run=False,
        sample=0,
        time_budget_min=0,
        workers=2,
        json=False,
        resume=False,
        verbose=False,
        debug=False,
        max_ram_mb=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_debug_flag_in_context(tmp_path, monkeypatch):
    """Test that debug flag is properly passed to context."""
    args = _namespace(tmp_path, debug=True)
    monkeypatch.setattr(builder_cli.config, "load_env", lambda: {})
    ctx = builder_cli.build_context(args, {"output": {"lang": "fr"}})
    assert ctx["debug"] is True


def test_verbose_flag_in_context(tmp_path, monkeypatch):
    """Test that verbose flag is properly passed to context."""
    args = _namespace(tmp_path, verbose=True)
    monkeypatch.setattr(builder_cli.config, "load_env", lambda: {})
    ctx = builder_cli.build_context(args, {"output": {"lang": "fr"}})
    assert ctx["verbose"] is True


def test_pipeline_configure_logging_debug():
    """Test that configure_logging properly handles debug mode."""
    logger = pipeline.configure_logging(verbose=False, debug=True)
    assert logger.level == logging.INFO
    

def test_pipeline_configure_logging_verbose():
    """Test that configure_logging properly handles verbose mode."""
    logger = pipeline.configure_logging(verbose=True, debug=False)
    assert logger.level == logging.DEBUG
    

def test_pipeline_configure_logging_both():
    """Test that verbose takes precedence over debug."""
    logger = pipeline.configure_logging(verbose=True, debug=True)
    assert logger.level == logging.DEBUG


def test_pipeline_configure_logging_none():
    """Test that no flags result in WARNING level."""
    logger = pipeline.configure_logging(verbose=False, debug=False)
    assert logger.level == logging.WARNING


@patch('builder_cli.pipeline.configure_logging')
def test_main_calls_configure_logging_with_debug(mock_configure_logging, tmp_path, monkeypatch):
    """Test that main() calls configure_logging with debug flag."""
    # Create a minimal job file
    job_file = tmp_path / "test_job.yaml"
    job_file.write_text("niche: test\nprofile: quick\n")
    
    # Mock sys.argv
    test_args = [
        'builder_cli.py', 'run-profile',
        '--job', str(job_file),
        '--out', str(tmp_path / 'out'),
        '--profile', 'quick',
        '--debug',
        '--explain'
    ]
    
    # Mock configure_logging to return a logger
    mock_logger = MagicMock()
    mock_configure_logging.return_value = mock_logger
    
    with patch('sys.argv', test_args):
        builder_cli.main()
    
    # Verify configure_logging was called with debug=True
    mock_configure_logging.assert_called_with(False, True)  # verbose=False, debug=True


@patch('builder_cli.pipeline.configure_logging')
def test_main_calls_configure_logging_with_verbose(mock_configure_logging, tmp_path, monkeypatch):
    """Test that main() calls configure_logging with verbose flag."""
    # Create a minimal job file
    job_file = tmp_path / "test_job.yaml"
    job_file.write_text("niche: test\nprofile: quick\n")
    
    # Mock sys.argv
    test_args = [
        'builder_cli.py', 'run-profile',
        '--job', str(job_file),
        '--out', str(tmp_path / 'out'),
        '--profile', 'quick',
        '--verbose',
        '--explain'
    ]
    
    # Mock configure_logging to return a logger
    mock_logger = MagicMock()
    mock_configure_logging.return_value = mock_logger
    
    with patch('sys.argv', test_args):
        builder_cli.main()
    
    # Verify configure_logging was called with verbose=True
    mock_configure_logging.assert_called_with(True, False)  # verbose=True, debug=False