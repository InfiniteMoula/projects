#!/usr/bin/env python3
"""Tests for job generation and batch processing functionality."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

import create_job
import builder_cli


def test_generate_niche_name():
    """Test NAF code to niche name conversion."""
    assert create_job.generate_niche_name("6920Z") == "naf_6920Z"
    assert create_job.generate_niche_name("43.29A") == "naf_4329A"
    assert create_job.generate_niche_name("43 . 29 A") == "naf_4329A"


def test_generate_job_config():
    """Test job configuration generation from template."""
    template = """
niche: "{niche_name}"
filters:
  naf_include: ["{naf_code}"]
profile: "{profile}"
output:
  dir: "out/{niche_name}"
"""
    result = create_job.generate_job_config("6920Z", template, "standard")
    config = yaml.safe_load(result)
    
    assert config["niche"] == "naf_6920Z"
    assert config["filters"]["naf_include"] == ["6920Z"]
    assert config["profile"] == "standard"
    assert config["output"]["dir"] == "out/naf_6920Z"


def test_create_job_file(tmp_path):
    """Test creating a job file from template."""
    template_path = tmp_path / "template.yaml"
    template_path.write_text("""
niche: "{niche_name}"
filters:
  naf_include: ["{naf_code}"]
profile: "{profile}"
""")
    
    job_file = create_job.create_job_file(
        "6920Z", tmp_path, template_path, "quick"
    )
    
    assert job_file.exists()
    assert job_file.name == "naf_6920Z.yaml"
    
    config = yaml.safe_load(job_file.read_text())
    assert config["niche"] == "naf_6920Z"
    assert config["filters"]["naf_include"] == ["6920Z"]
    assert config["profile"] == "quick"


def test_generate_batch_jobs(tmp_path):
    """Test generating multiple job files."""
    template_path = tmp_path / "template.yaml"
    template_path.write_text("""
niche: "{niche_name}"
filters:
  naf_include: ["{naf_code}"]
profile: "{profile}"
""")
    
    naf_codes = ["6920Z", "4329A", "43"]
    job_files = create_job.generate_batch_jobs(
        naf_codes, tmp_path, template_path, "standard"
    )
    
    assert len(job_files) == 3
    assert all(f.exists() for f in job_files)
    
    # Check each generated file
    for i, job_file in enumerate(job_files):
        config = yaml.safe_load(job_file.read_text())
        expected_naf = naf_codes[i]
        expected_niche = f"naf_{expected_naf.replace('.', '').replace(' ', '').upper()}"
        
        assert config["niche"] == expected_niche
        assert config["filters"]["naf_include"] == [expected_naf]
        assert config["profile"] == "standard"


def test_batch_jobs_with_default_template(tmp_path):
    """Test batch job generation with default template."""
    # Copy the default template to a temporary location
    default_template = Path(__file__).parent.parent / "job_template.yaml"
    if default_template.exists():
        template_content = default_template.read_text()
        test_template = tmp_path / "job_template.yaml"
        test_template.write_text(template_content)
        
        # Generate jobs with default template
        naf_codes = ["6920Z", "4329A"]
        job_files = create_job.generate_batch_jobs(
            naf_codes, tmp_path, test_template, "quick"
        )
        
        assert len(job_files) == 2
        assert all(f.exists() for f in job_files)
        
        # Verify the content is valid YAML and has the expected structure
        for job_file in job_files:
            config = yaml.safe_load(job_file.read_text())
            assert "niche" in config
            assert "filters" in config
            assert "naf_include" in config["filters"]
            assert "profile" in config
            assert config["profile"] == "quick"


def test_run_function_returns_skipped():
    """Test that the run function returns SKIPPED status."""
    result = create_job.run({}, {})
    assert result["status"] == "SKIPPED"
    assert "message" in result


class TestBatchCLIArgs:
    """Test batch command argument parsing."""
    
    def test_batch_command_exists(self):
        """Test that batch command is available."""
        import subprocess
        result = subprocess.run(
            ["python", "builder_cli.py", "batch", "--help"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "NAF code(s)" in result.stdout
        assert "--dry-run" in result.stdout
        assert "--continue-on-error" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__])