"""Airflow DAG factory mirroring the builder pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator

from builder_cli import PROFILES, STEP_DEPENDENCIES
from .runner import run_step_cli


def _python_callable_factory(
    step_name: str,
    job_path: str,
    outdir: str,
    extra_args: Optional[Iterable[str]] = None,
    env: Optional[Dict[str, str]] = None,
):
    def _callable(**_kwargs):
        result = run_step_cli(
            step_name,
            job_path,
            outdir,
            extra_args=extra_args,
            env=env,
            capture_output=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"builder step {step_name} exited with {result.returncode}")

    return _callable


def create_builder_dag(
    dag_id: str,
    job_path: str,
    outdir: str,
    *,
    profile: str = "standard",
    schedule: Optional[str] = None,
    extra_args: Optional[Iterable[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> DAG:
    """
    Build an Airflow DAG aligned with the canonical pipeline profile.
    """

    steps = PROFILES.get(profile)
    if not steps:
        raise ValueError(f"Unknown profile '{profile}'. Available: {sorted(PROFILES)}")

    dag = DAG(
        dag_id=dag_id,
        description=f"Builder pipeline profile '{profile}'",
        start_date=datetime(2024, 1, 1),
        schedule_interval=schedule,
        catchup=False,
        max_active_runs=1,
        default_args={
            "owner": "builder",
            "depends_on_past": False,
            "retries": 1,
        },
    )

    airflow_tasks: Dict[str, PythonOperator] = {}
    with dag:
        for step in steps:
            task = PythonOperator(
                task_id=step.replace(".", "__"),
                python_callable=_python_callable_factory(
                    step, job_path, outdir, extra_args=extra_args, env=env
                ),
            )
            airflow_tasks[step] = task

        for step, task in airflow_tasks.items():
            for dep in STEP_DEPENDENCIES.get(step, set()):
                if dep in airflow_tasks:
                    airflow_tasks[dep] >> task

    return dag


# Example DAG registration for the default profile
builder_standard_dag = create_builder_dag(
    "builder_standard",
    job_path="/opt/pipeline/job.yaml",
    outdir="/opt/pipeline/out",
    profile="standard",
)


__all__ = ["create_builder_dag", "builder_standard_dag"]
