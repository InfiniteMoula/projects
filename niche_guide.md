python create_job.py jobs/my_niche.yaml

Édite naf_include, regions, budgets.

python builder_cli.py run-profile --job jobs/my_niche.yaml --input data/sirene_latest.parquet --out out/my_niche --dry-run --sample 50

Si OK, retire --dry-run, passe en standard.

Deep si besoins PDF/HTML.