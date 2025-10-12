"""Prefect flow factory mapped to the canonical builder pipeline."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

from prefect import flow, task, get_run_logger

from builder_cli import PROFILES, STEP_DEPENDENCIES

from .runner import run_step_cli


@task(name="builder-step", retries=0)
def run_step_task(
    step_name: str,
    job_path: str,
    outdir: str,
    extra_args: Optional[Iterable[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """
    Execute a single builder pipeline step via the CLI.

    The task mirrors the behaviour of ``python builder_cli.py run-step`` and
    returns the captured stdout to support downstream logging or artefact
    persistence within Prefect.
    """

    logger = get_run_logger()
    logger.info("Running step %s", step_name)
    result = run_step_cli(
        step_name,
        job_path,
        outdir,
        extra_args=extra_args,
        env=env,
        capture_output=True,
    )

    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.warning(result.stderr.strip())

    if result.returncode != 0:
        raise RuntimeError(
            f"Step {step_name} exited with status {result.returncode}. "
            f"Command: {' '.join(result.args)}"
        )
    return result.stdout


@flow(name="builder-prefect-flow")
def builder_prefect_flow(
    job_path: str,
    outdir: str,
    profile: str = "standard",
    *,
    extra_cli_args: Optional[Iterable[str]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """
    Orchestrate the canonical builder pipeline with Prefect.

    Parameters
    ----------
    job_path:
        Path to the job configuration YAML.
    outdir:
        Output directory for builder artefacts.
    profile:
        Pipeline profile to execute (defaults to ``standard``).
    extra_cli_args:
        Optional iterable of command line flags forwarded to each
        ``builder_cli`` invocation.
    env:
        Optional mapping of environment variables applied to every task.
    """

    logger = get_run_logger()
    steps = PROFILES.get(profile)
    if not steps:
        raise ValueError(f"Unknown profile '{profile}'. Available: {sorted(PROFILES)}")

    logger.info("Launching Prefect flow for profile '%s' (%d steps)", profile, len(steps))

    results = {}
    futures = {}
    for step in steps:
        deps = [
            futures[dep]
            for dep in STEP_DEPENDENCIES.get(step, set())
            if dep in futures
        ]
        future = run_step_task.submit(
            step,
            job_path,
            outdir,
            extra_cli_args,
            env,
            wait_for=deps,
        )
        futures[step] = future

    for step, future in futures.items():
        results[step] = future.result()
    return results


__all__ = ["builder_prefect_flow", "run_step_task"]
