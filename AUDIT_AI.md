# Repository Knowledge Graph (AI-Friendly)

## metadata
- repo_name: projects
- primary_language: Python 3.11
- domain: B2B data collection, enrichment, quality assurance
- entry_point: builder_cli.py
- cli_commands: run-profile, run-step, resume, batch
- key_configs: config/enrichment.yaml, job_template.yaml, config/budget_config.py
- automation_targets: Prefect, Dagster, Airflow, Prometheus/Grafana

## global_workflow
1. collect: dumps.collect, api.collect, http.serp, crawl.site*, headless.collect* (nethttp/, headless/, feeds/, pdf/, api/)
2. parse_normalize: parse.html/jsonld/pdf, normalize.standardize (parse/, normalize/)
3. enrich: enrich.address/domain/site/dns/email/phone, enrich.contacts/linkedin, scraper.maps, correlation.checks (enrich/, scraper/, correlation/)
4. quality_package: quality.* (checks/dedupe/score/clean_contacts), package.export, finalize.premium_dataset (quality/, package/)
5. observe: metrics.export, Prometheus exporter, reports/report_metrics.json (metrics/, monitoring/)

## core_files
- builder_cli.py: CLI orchestrator (step registry, dependencies, profiles, budgets, KPI, Prometheus hooks, parallel execution, resume logic).
- create_job.py: Generates job YAML from templates/NAF codes (used by batch mode).
- job_template.yaml: Parameterized blueprint for jobs (filters, profiles, budgets, KPI, scraper settings).
- README.md: Human-readable overview (architecture, commands, configuration, observability, testing).
- AUDIT.md: Narrative audit for humans (sections 1–13).
- AUDIT_AI.md: This structured document for machine ingestion.

## directories
- path: .github
  type: ci_config
  description: GitHub Actions workflows (CI, coverage badge) and badges.
- path: .pytest_cache / __pycache__ / .venv / venv / out
  type: generated
  description: Runtime artefacts (test cache, bytecode, virtual envs, pipeline outputs). Should be ignored by tooling.
- path: ai
  type: package
  description: AI helpers (`extract_llm.py`) for LLM-based extraction or enrichment experiments.
- path: api
  type: package
  description: API clients/collectors for external B2B sources (called from `STEP_REGISTRY` via `api.collect`).
- path: benchmarks
  type: experiments
  description: Benchmark scripts/notebooks for performance comparisons (e.g., crawl throughput, ML).
- path: cache
  type: storage
  description: Local caches (HTTP, SERP, ML artefacts) used during enrichment; managed via config/enrichment.yaml.
- path: collect
  type: package
  description: Helpers such as `nafreference.py` for NAF lookups and common collection utilities.
- path: compliance
  type: docs_and_scripts
  description: Compliance/legal workflows (e.g., GDPR handling); complements `docs/legal.md`.
- path: config
  type: package
  description: Configuration loaders (`enrichment_config.py`, `budget_config.py`, env helpers) and YAML defaults.
- path: constants
  type: data
  description: Reference lists (generic domains/emails, regexes) consumed by enrich/quality modules.
- path: correlation
  type: package
  description: Correlation/scoring logic between enriched signals (`correlation/correlation.py` step).
- path: dashboard
  type: ui_assets
  description: Dashboard definitions (marketing/ops visualizations) built from exported datasets.
- path: data
  type: samples
  description: Sample datasets (e.g., `sirene_sample.parquet`, user agent pools) for testing/prototyping.
- path: docs
  type: documentation
  description: Extended guides (enrichment, orchestration, observability, scraping, data dictionaries, legal, upgrade notes).
- path: dumps
  type: package
  description: Modules to ingest SIRENE dumps or other bulk datasets (`dumps.collect_dump:run` etc.).
- path: enrich
  type: package
  description: Enrichment modules (domains, contacts, LinkedIn, Google Maps, DNS, phone/email heuristics).
- path: examples
  type: reference
  description: Example configs/scripts demonstrating how to run specific jobs or analyses.
- path: features
  type: research
  description: Feature engineering experiments or future pipeline capabilities.
- path: feeds
  type: package
  description: RSS/atom ingestion steps invoked via `feeds.collect`.
- path: headless
  type: package
  description: Playwright-based crawlers (`headless.collect*`) and fallbacks for dynamic websites.
- path: jobs
  type: configs
  description: Pre-built job YAMLs per NAF (`naf_*.yaml`) plus marketing/internal presets.
- path: marketing_exports
  type: scripts
  description: Tools to transform pipeline outputs into marketing-friendly deliverables (`export_marketing.py`, `generate_professional_services_final.py`).
- path: metrics
  type: package
  description: Metrics aggregation/export (`metrics/collector.py`, `metrics/exporter.py`).
- path: ml
  type: package
  description: Machine learning models (address classifier, extraction models, domain predictor) and training scripts.
- path: monitor / monitoring
  type: observability
  description: Monitoring utilities plus Docker Compose stack for Prometheus/Grafana deployment.
- path: net / nethttp
  type: package
  description: Networking layer (HTTP clients, SERP handling, sitemap discovery, crawl pipelines).
- path: normalize
  type: package
  description: Standardization/normalization logic for SIRENE dumps and other inputs (`normalize/standardize.py`).
- path: orchestration
  type: integrations
  description: Prefect, Dagster, Airflow integration layers reusing the CLI registry.
- path: package
  type: package
  description: `package/exporter.py` and helpers to merge normalized data with enriched contacts and metadata.
- path: parse
  type: package
  description: Parsers for HTML, JSON-LD, PDF, contacts (including AI pipelines).
- path: pdf
  type: package
  description: PDF download and parsing steps referenced by `parse.pdf`.
- path: quality
  type: package
  description: Quality framework: checks, scoring, deduplication, contact cleaning, validation utilities (`QUALITY_FRAMEWORK.md`).
- path: scraper
  type: package
  description: Specialized scrapers (e.g., Google Maps via `scraper/maps_scraper.py`) outside core nethttp logic.
- path: scripts
  type: tools
  description: Helper scripts for maintenance, migrations or one-off automation tasks.
- path: serp
  type: package
  description: SERP provider implementations/config for domain/linkedin discovery.
- path: tests
  type: tests
  description: Pytest suite covering budgets, CLI, enrichment, parsing, quality, exporters, adaptive logic, AI fallback.
- path: utils
  type: package
  description: Shared utilities (budget middleware, adaptive controller, directories, IO, logging filters, config helpers).

## notable_top_level_files
- ENRICHMENT_IMPLEMENTATION.md: Deep dive into enrichment modules and zero-capital workflow.
- QUALITY_FRAMEWORK.md: Governance around quality metrics and acceptance criteria.
- requirements.txt: Python dependencies (pandas, pyarrow, httpx, playwright, scikit-learn, etc.).
- pyproject.toml: Tooling config (Black, Ruff, MyPy, Pytest path).
- .pre-commit-config.yaml: Linters/formatters used before commits.
- demo_phase4_automation.py / demo_quality_framework.py / test_phase3*.py: Legacy demos/tests for phase-based deployments.
- train_domain_predictor.py: Script to train ML models predicting domains.
- tmp_linkedin.html: Sample HTML used in parsing/linkedin tests.
- builder_cli.py / create_job.py / README.md / AUDIT.md / AUDIT_AI.md: Core entrypoints and documentation.

## operational_notes
- Environment variables: load via `config.load_env()`; `.env.example` lists required keys (API tokens, proxies, budgets).
- Outputs: All pipeline runs write under `out/<run>/` with subfolders normalized/, contacts/, logs/, reports/, metrics/. These directories are auto-created by `utils.directories.initialize_pipeline_directories`.
- Budgets/KPI: configurable per job; enforced during each step with graceful errors (`BudgetTracker`, `KPICalculator`).
- Observability: enable via `--prometheus-port`; optional Grafana stack in `monitoring/`.
- Tests: run `pytest`; lint via `ruff check .` and `black .`.

