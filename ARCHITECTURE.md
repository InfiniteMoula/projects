# Architecture

## Pipeline Overview
- `builder_cli.py` centralises step orchestration. It resolves step names via `STEP_REGISTRY`, orders them with `STEP_DEPENDENCIES`, and executes batches according to the selected `profile`. Every step receives the whole job configuration plus a runtime `ctx` (outdir paths, logging, budgets, resume metadata, etc.).
- Export-focused jobs (like `top_500_premium`) reuse the same orchestration surface but only execute lightweight Pandas transformations. They do not interfere with crawling or enrichment modules already present in the repository.

## Key Modules
- `builder_cli.py` — maintains the canonical registry of step callables, dependency graph, feature flags, and profile definitions. It now exposes a `top_500_premium` profile composed of six deterministic steps dedicated to premium dataset generation.
- `pipeline/top_500_premium.py` — holds reusable dataframe helpers (`filter_rows`, `sort_values`, `head`, `reorder_columns`) plus the concrete step implementations:
  - `load_enriched_dataset`: loads `data/enriched/all_companies_enriched.parquet` (or the CLI `--input`) into the shared context.
  - `filter_sellable`: enforces the `is_sellable == True` filter.
  - `ensure_score`: guarantees the `score_global` column exists and fills missing scores with `0.0`.
  - `sort_by_score`, `select_head`, `reorder_for_export`, `export_csv`: handle ranking, slicing, column ordering, and writing `exports/top_500_premium.csv` with UTF-8-SIG encoding. Each step logs progress and records small summaries for observability.

## Job Definitions
- `jobs/top_500_premium.yaml`
  - Declares `profile: top_500_premium`, so running `builder_cli.py run-profile --job jobs/top_500_premium.yaml --profile top_500_premium --out out/top_500_premium` will execute the dataset pipeline sequentially.
  - The `top_500_premium` block configures IO (input parquet, `exports/top_500_premium.csv` target, encoding), filtering expression, score defaults, limit (500 rows) and the column ordering list. This keeps the business rules co-located with the job spec for easy tweaks without touching code.
  - The generic `output` section still references a run directory for logs/KPIs, matching the existing job pattern.

## Workflow Impact
- Input: `data/enriched/all_companies_enriched.parquet` (already enriched upstream by standard jobs).
- Steps executed (new profile):
  1. `top500.load_dataset`
  2. `top500.filter_sellable`
  3. `top500.ensure_score`
  4. `top500.sort_score`
  5. `top500.limit`
  6. `top500.reorder`
  7. `top500.export_csv`
- Output: `exports/top_500_premium.csv` (UTF-8-SIG, 500 best `score_global` rows, Excel-friendly column order). The dataframe copy is also stored inside the run context for downstream tooling or reporting.

## Recent Changes
- Added `pipeline/top_500_premium.py` with shared dataframe helpers and six atomic steps dedicated to premium selection/export logic.
- Registered new step names plus the `top_500_premium` profile inside `builder_cli.py`.
- Created `jobs/top_500_premium.yaml`, binding business constraints (filters, score defaults, ordering) to the reusable pipeline.
