# Niche Development Guide

## Individual Job Creation

python create_job.py jobs/my_niche.yaml

Édite naf_include, regions, budgets.

python builder_cli.py run-profile --job jobs/my_niche.yaml --input data/sirene_latest.parquet --out out/my_niche --dry-run --sample 50

Si OK, retire --dry-run, passe en standard.

Deep si besoins PDF/HTML.

## Batch Processing (NEW)

For processing multiple NAF codes efficiently:

```bash
# Generate and run jobs for multiple NAF codes
python builder_cli.py batch --naf 6920Z --naf 4329A --naf 43 --input data/sirene_latest.parquet --output-dir out/batch_results

# Dry run to generate jobs only
python builder_cli.py batch --naf 6920Z --naf 4329A --input data/sirene_latest.parquet --output-dir out/batch_results --dry-run

# With error tolerance
python builder_cli.py batch --naf 6920Z --naf 4329A --input data/sirene_latest.parquet --output-dir out/batch_results --continue-on-error
```

See BATCH_GUIDE.md for complete documentation.