"""Dagster job definition mirroring the builder pipeline."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from dagster import In, Nothing, OpExecutionContext, job, op

from builder_cli import PROFILES, STEP_DEPENDENCIES
from .runner import run_step_cli


def _make_ins(step_name: str) -> Dict[str, In]:
    return {
        dep.replace(".", "__"): In(Nothing)
        for dep in STEP_DEPENDENCIES.get(step_name, set())
    }


def _make_step_op(step_name: str):
    ins = _make_ins(step_name)

    @op(
        name=step_name.replace(".", "__"),
        ins=ins,
        config_schema={
            "job_path": str,
            "outdir": str,
            "extra_args": [str],
        },
    )
    def _run_step(context: OpExecutionContext) -> str:
        cfg = context.op_config
        extra_args: Optional[Iterable[str]] = cfg.get("extra_args")
        result = run_step_cli(
            step_name,
            cfg["job_path"],
            cfg["outdir"],
            extra_args=extra_args,
            env=None,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Step {step_name} failed with exit code {result.returncode}.\n"
                f"Stdout: {result.stdout}\nStderr: {result.stderr}"
            )
        return result.stdout

    return _run_step


def create_builder_job(profile: str):
    """
    Return a Dagster ``JobDefinition`` for the requested pipeline profile.
    """

    steps = PROFILES.get(profile)
    if not steps:
        raise ValueError(f"Unknown profile '{profile}'. Available: {sorted(PROFILES)}")

    ops = {step: _make_step_op(step) for step in steps}

    @job(name=f"builder_{profile}_job", description=f"Dagster mapping for builder profile '{profile}'")
    def _builder_job():
        op_results: Dict[str, object] = {}
        for step in steps:
            op_fn = ops[step]
            upstream = {
                dep.replace(".", "__"): op_results[dep]
                for dep in STEP_DEPENDENCIES.get(step, set())
                if dep in op_results
            }
            op_results[step] = op_fn(**upstream)

    return _builder_job


# Convenience alias for registration (default profile).
builder_standard_job = create_builder_job("standard")


__all__ = ["create_builder_job", "builder_standard_job"]
