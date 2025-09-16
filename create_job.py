#!/usr/bin/env python3
"""Job generator for creating jobs from NAF codes."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import yaml

from utils import io


def generate_niche_name(naf_code: str) -> str:
    """Generate a meaningful niche name from NAF code."""
    naf_clean = naf_code.replace(".", "").replace(" ", "").upper()
    return f"naf_{naf_clean}"


def load_template(template_path: Path) -> str:
    """Load job template from file."""
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def generate_job_config(
    naf_code: str,
    template_content: str,
    profile: str = "quick",
    custom_params: Dict[str, str] = None
) -> str:
    """Generate a job configuration from template and NAF code."""
    niche_name = generate_niche_name(naf_code)
    
    # Default substitution parameters
    params = {
        "naf_code": naf_code,
        "niche_name": niche_name,
        "profile": profile,
    }
    
    # Add custom parameters if provided
    if custom_params:
        params.update(custom_params)
    
    # Replace template variables
    job_content = template_content.format(**params)
    
    return job_content


def create_job_file(
    naf_code: str,
    output_dir: Path,
    template_path: Path,
    profile: str = "quick",
    custom_params: Dict[str, str] = None
) -> Path:
    """Create a job file for a given NAF code."""
    template_content = load_template(template_path)
    job_content = generate_job_config(naf_code, template_content, profile, custom_params)
    
    niche_name = generate_niche_name(naf_code)
    job_file = output_dir / f"{niche_name}.yaml"
    
    io.write_text(job_file, job_content)
    return job_file


def generate_batch_jobs(
    naf_codes: List[str],
    output_dir: Path,
    template_path: Path = None,
    profile: str = "quick",
    custom_params: Dict[str, str] = None
) -> List[Path]:
    """Generate job files for multiple NAF codes."""
    if template_path is None:
        template_path = Path(__file__).parent / "job_template.yaml"
    
    job_files = []
    for naf_code in naf_codes:
        try:
            job_file = create_job_file(naf_code, output_dir, template_path, profile, custom_params)
            job_files.append(job_file)
            print(f"Generated job file: {job_file}")
        except Exception as exc:
            print(f"Failed to generate job for NAF {naf_code}: {exc}")
    
    return job_files


def run(cfg: dict, ctx: dict) -> dict:
    """Run function for pipeline compatibility."""
    # This is for pipeline compatibility, but the main functionality
    # is accessed through the CLI batch command
    return {"status": "SKIPPED", "message": "Use 'builder_cli batch' command for job generation"}


def main():
    """Main CLI entry point for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Generate job files from NAF codes",
        prog="create_job.py"
    )
    parser.add_argument(
        "output_path",
        help="Output path for the job file (for single job) or directory (for batch jobs)"
    )
    parser.add_argument(
        "--naf", "--naf-code",
        dest="naf_codes",
        action="append",
        help="NAF code(s) to generate jobs for (can be used multiple times)"
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="Path to job template file (default: job_template.yaml)"
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "standard", "deep"],
        default="quick",
        help="Profile to use for the job (default: quick)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Generate multiple job files (one per NAF code)"
    )
    
    args = parser.parse_args()
    
    if not args.naf_codes:
        parser.error("At least one NAF code must be specified with --naf")
    
    output_path = Path(args.output_path)
    
    if args.batch:
        # Batch mode: create multiple job files
        io.ensure_dir(output_path)
        job_files = generate_batch_jobs(
            args.naf_codes,
            output_path,
            args.template,
            args.profile
        )
        print(f"Generated {len(job_files)} job files in {output_path}")
    else:
        # Single mode: create one job file (use first NAF code)
        if len(args.naf_codes) > 1:
            print(f"Warning: Multiple NAF codes provided, using only the first one: {args.naf_codes[0]}")
        
        template_path = args.template or (Path(__file__).parent / "job_template.yaml")
        create_job_file(args.naf_codes[0], output_path.parent, template_path, args.profile)
        print(f"Generated job file: {output_path}")


if __name__ == "__main__":
    main()
