"""Shared helpers for orchestrator integrations."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence


def ensure_iterable(values: Optional[Iterable[str]]) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return list(values)
    if isinstance(values, str):
        return [values]
    return list(values)


def build_cli_command(
    step_name: str,
    job_path: Path,
    outdir: Path,
    extra_args: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Compose the ``builder_cli`` invocation for a single step.
    """

    cmd: List[str] = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "builder_cli.py"),
        "run-step",
        "--job",
        str(job_path),
        "--out",
        str(outdir),
        "--step",
        step_name,
        "--resume",
    ]
    cmd.extend(ensure_iterable(extra_args))
    return cmd


def run_step_cli(
    step_name: str,
    job_path: str,
    outdir: str,
    *,
    extra_args: Optional[Iterable[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[Sequence[str]]:
    """
    Execute a builder step via the CLI and return the ``CompletedProcess``.
    """

    job_file = Path(job_path).expanduser().resolve()
    out_dir = Path(outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env_vars: MutableMapping[str, str] = dict(os.environ)
    if env:
        env_vars.update(env)

    return subprocess.run(
        build_cli_command(step_name, job_file, out_dir, extra_args),
        check=False,
        capture_output=capture_output,
        text=True,
        env=env_vars,
    )


__all__ = ["build_cli_command", "run_step_cli", "ensure_iterable"]
