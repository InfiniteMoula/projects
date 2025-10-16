"""End-to-end integration tests for pipeline profiles."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

import builder_cli
from tests import profile_step_stubs


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    """Provide mandatory secrets so the pipeline passes validation."""
    monkeypatch.setattr(
        builder_cli.config,
        "load_env",
        lambda: {
            "APIFY_API_TOKEN": "test-token",
            "HUNTER_API_KEY": "test-key",
            "HTTP_PROXY": "http://localhost",
        },
    )


@pytest.fixture
def stubbed_registry(monkeypatch):
    """Patch the step registry with fast stub implementations."""
    registry = dict(builder_cli.STEP_REGISTRY)
    registry.update(profile_step_stubs.build_registry())
    monkeypatch.setattr(builder_cli, "STEP_REGISTRY", registry)
    yield registry


def _run_profile(tmp_path: Path, profile: str) -> List[str]:
    """Execute the CLI for *profile* and return the executed steps order."""
    profile_step_stubs.reset()
    outdir = tmp_path / "out"
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        "\n".join(
            [
                f"niche: test-{profile}",
                f"profile: {profile}",
                "filters:",
                "  naf_include: ['6202A']",
                "output:",
                f"  dir: '{outdir.as_posix()}'",
            ]
        )
    )
    input_file = tmp_path / "input.parquet"
    input_file.write_text("placeholder")

    args = [
        "builder_cli.py",
        "run-profile",
        "--job",
        str(job_file),
        "--out",
        str(outdir),
        "--input",
        str(input_file),
        "--profile",
        profile,
    ]

    with patch.object(sys, "argv", args):
        exit_code = builder_cli.main()

    assert exit_code == 0
    return profile_step_stubs.executions()


@pytest.mark.parametrize("profile", ["quick", "standard", "deep"])
def test_profile_execution_order(tmp_path, stubbed_registry, profile):
    """Each predefined profile should execute all steps in dependency order."""
    executed_steps = _run_profile(tmp_path, profile)
    batches = builder_cli.build_execution_batches(builder_cli.PROFILES[profile])
    expected_order = [step for batch in batches for step in batch]
    assert executed_steps == expected_order

    outdir = tmp_path / "out"
    assert (outdir / "dataset.csv").exists()
    assert (outdir / "dataset.parquet").exists()

