# Orchestration Blueprint

This document captures the mapping between the canonical `builder_cli` pipeline
and the orchestration artefacts delivered in `orchestration/`.  It is intended
as a starting point for running the end‑to‑end workflow under Prefect, Dagster
or Airflow while keeping the original step registry and dependency graph.

## Step Topology

| Phase                    | Representative steps (`builder_cli` name)                               | Blocking dependencies                    |
|--------------------------|---------------------------------------------------------------------------|------------------------------------------|
| Collection               | `dumps.collect`, `api.collect`, `http.static`, `http.sitemap`, `http.serp` | `http.static` → `dumps.collect`; `http.serp` → `normalize.standardize`, `enrich.address` |
| Crawling                 | `crawl.site`, `crawl.site_async`, `headless.collect*`                    | `crawl.*` → `http.serp`; `headless.collect` → `dumps.collect` |
| Parsing & Normalisation  | `parse.html`, `parse.jsonld`, `normalize.standardize`                    | Parsing steps gate `normalize.standardize` |
| Enrichment               | `enrich.address`, `enrich.google_maps`, `enrich.domain`, `enrich.site`, `enrich.email`, `enrich.phone` | `enrich.address` → `normalize.standardize`; `enrich.google_maps` ↔ other enrich.* |
| Quality & Packaging      | `quality.checks`, `quality.score`, `package.export`                      | Quality checks depend on normalisation and enrichment; packaging is last |

> ℹ️  The precise dependency map is expressed in `builder_cli.STEP_DEPENDENCIES`
> and imported dynamically by the orchestration code to avoid drift.

## Prefect

* Module: `orchestration/prefect_flow.py`
* Entry point: `builder_prefect_flow(job_path, outdir, profile="standard", extra_cli_args=None, env=None)`
* Behaviour:
  - Builds a dynamic task graph based on the selected profile.
  - Each task invokes `builder_cli.py run-step …` with `--resume`, ensuring restartability.
  - Prefect waits on dependencies as defined in `STEP_DEPENDENCIES`.
* Retry strategy: handled by Prefect’s flow runtime (`run_step_task` defaults to zero retries so that Prefect-level retry policies can be attached per deployment).
* Failure tolerance: the flow stops on the first failing step which mirrors current CLI semantics. Prefect deployments can override this by wrapping the flow in a higher‑level orchestration with custom retry/backoff logic.

### Example deployment snippet

```python
from orchestration.prefect_flow import builder_prefect_flow

builder_prefect_flow.deploy(
    name="builder-standard",
    parameters={
        "job_path": "/data/jobs/standard.yaml",
        "outdir": "/data/out",
        "profile": "standard",
        "extra_cli_args": ["--prometheus-port", "9099"],
    },
    work_queue_name="builder-workers",
)
```

## Dagster

* Module: `orchestration/dagster_job.py`
* Factory: `create_builder_job(profile: str)` returning a `JobDefinition`.
* Default job: `builder_standard_job`.
* Execution:
  - Every Dagster op equates to a pipeline step and expects the following `op_config` keys: `job_path`, `outdir`, and optional `extra_args`.
  - Dependencies are wired through `Nothing` inputs according to `STEP_DEPENDENCIES`.
* Failure tolerance: the job aborts on first failing op. Use Dagster run retries (e.g. `RetryPolicy`) for automated reruns.
* Sample run configuration:

```yaml
ops:
  dumps_collect:
    config:
      job_path: /data/jobs/standard.yaml
      outdir: /data/out
      extra_args: ["--prometheus-port", "9099"]
  api_collect:
    config:
      job_path: /data/jobs/standard.yaml
      outdir: /data/out
      extra_args: ["--prometheus-port", "9099"]
# ... repeat for all ops or leverage presets/helpers
```

> Tip: wrap `create_builder_job` in a Dagster repository and attach a shared config resource to avoid repeating `job_path`/`outdir` per op.

## Airflow

* Module: `orchestration/airflow_dag.py`
* Factory: `create_builder_dag(dag_id, job_path, outdir, profile="standard", schedule=None, extra_args=None, env=None)`
* Default DAG: `builder_standard_dag` (registered with example paths).
* Failure tolerance: each `PythonOperator` inherits Airflow’s default retry once behaviour (`retries=1`). Adjust via DAG `default_args`.
* Backfill/catchup: disabled (`catchup=False`) given the pipeline focuses on near real-time enrichment.

### Alerting

- Configure task-level email/webhook alerts via Airflow `on_failure_callback` to surface failed steps quickly.
- Airflow SLA/missed schedule warnings provide early signals when upstream data sources slow down.

## Retry & Failure Policy Summary

| Platform | Default retries | Recommendation                            |
|----------|-----------------|--------------------------------------------|
| Prefect  | 0 (per task)    | Use deployment-level retry/backoff (e.g. exponential with jitter) and mark idempotent steps as retryable. |
| Dagster  | 0               | Attach `RetryPolicy(max_retries=2, delay=60)` to network-heavy ops (`http.*`, `enrich.*`). |
| Airflow  | 1               | Increase to 2–3 retries for SERP/crawl tasks with `retry_exponential_backoff=True`. |

## Failure Tolerance Guidelines

1. **Transient HTTP failures**: Allow retries, leverage the built-in builder budget tracker to short-circuit on quota exhaustion.
2. **Data quality issues**: If a parsing or enrichment step fails deterministically, fail fast and surface the offending input through Prometheus (see observability doc) to avoid masking systemic issues.
3. **Downstream packaging**: `package.export` should be the only step with `depends_on_past=True` (Airflow) or `ins=[...]` (Dagster/Prefect) to ensure incomplete runs do not ship artefacts.

## Next Steps

1. Parameterise `job_path` and `outdir` through deployment variables/secrets in the orchestrator of your choice.
2. Combine with the Prometheus/Grafana stack (see `docs/observability.md`) to track throughput, success ratio and latency per step.
3. Extend Prefect/Dagster jobs with sensors to trigger reruns when new inputs land or budgets reset.
