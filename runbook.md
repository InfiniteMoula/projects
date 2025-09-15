# 1) bootstrap
powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
# 2) run quick profile on experts-comptables (input = local SIRENE .parquet or .csv)
python builder_cli.py run-profile --job jobs/experts_comptables.yaml --input data\sirene_latest.parquet --out out\ec_quick --sample 50 --dry-run
python builder_cli.py run-profile --job jobs/experts_comptables.yaml --input data\sirene_latest.parquet --out out\ec_std
# 3) package present in out\<run>\dataset.csv / dataset.parquet
