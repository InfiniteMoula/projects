# Observability & Metrics Export

This document explains how Prometheus/Grafana support is wired into the
pipeline and how to operate the bundled monitoring stack.

## Prometheus Exporter

* Module: `monitoring/prometheus_exporter.py`
* Enabled by passing `--prometheus-port <port>` (and optionally
  `--prometheus-address <addr>`) to any `builder_cli` command.
* Metrics exposed:
  - `pipeline_step_duration_seconds` (histogram): execution latency per step.
  - `pipeline_step_status_total` (counter): success/failure counts.
  - `pipeline_step_last_duration_seconds` (gauge): duration of the last run per step.
  - `pipeline_step_completed_timestamp_seconds` (gauge): Unix timestamp of the last completed run per step.
  - `pipeline_run_info` (gauge): high-level run metadata (run id, job and profile).

### Usage

```bash
python builder_cli.py run-profile \
  --job jobs/standard.yaml \
  --out out/run_2024_10_09 \
  --profile standard \
  --prometheus-port 9099
```

The exporter starts an HTTP endpoint on `0.0.0.0:9099/metrics`.

## Docker Compose Stack

The file `monitoring/docker-compose.yml` provisions:

| Service          | Description                                                    |
|------------------|----------------------------------------------------------------|
| `pipeline-worker`| Example container running `builder_cli` with Prometheus export |
| `prometheus`     | Scrapes `pipeline-worker:9099` every 15 seconds                |
| `grafana`        | Pre-provisioned datasource & dashboard for pipeline metrics    |

Bring the stack up:

```bash
cd monitoring
docker compose up -d
```

Prometheus UI: <http://localhost:9090>  
Grafana UI: <http://localhost:3000> (default credentials `admin`/`admin`)

## Grafana Dashboard & Alerts

* Dashboard provisioned from `monitoring/grafana/provisioning/dashboards/pipeline_overview.json`.
* Key visualisations:
  - Step duration percentiles (P50/P95).
  - Success vs failure counter per step.
  - Recent run timeline giving per-step completion timestamps.
* Suggested alerts:
  1. **Step stuck/failing** – trigger when `rate(pipeline_step_status_total{status="FAIL"}[5m]) > 0`.
  2. **Latency regression** – warn if P95 of `pipeline_step_duration_seconds` exceeds a run-specific threshold (configurable per profile).
  3. **No data** – use `absent(pipeline_step_completed_timestamp_seconds)` for core steps to detect exporter outages.

## Integration Notes

1. `prometheus_client` is now declared in `requirements.txt`. Install/update the environment before enabling the exporter.
2. The exporter is optional; when `prometheus_client` is missing or the port is set to `0`, the pipeline behaves as before.
3. The Grafana dashboard reads from the default Prometheus datasource (`prometheus`). Adjust `monitoring/grafana/provisioning/datasources/prometheus.yaml` to match custom deployments.
