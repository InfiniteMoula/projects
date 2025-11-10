# Pipeline de donnees B2B francaises

[![CI](https://github.com/infinitemoula/projects/actions/workflows/ci.yml/badge.svg)](https://github.com/infinitemoula/projects/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/infinitemoula/projects/main/.github/badges/coverage.json)](https://github.com/infinitemoula/projects)

Pipeline industriel utilise par l'equipe Infinitemoula pour collecter, normaliser, enrichir et empaqueter des jeux de donnees B2B issus de SIRENE et de sources web (SERP, crawling, Playwright). La CLI `builder_cli.py` orchestre plus de 40 etapes, applique des budgets/ KPI et expose des metriques exploitables via Prometheus ou JSON.

## Sommaire
- [Contexte et objectifs](#contexte-et-objectifs)
- [Architecture et flux](#architecture-et-flux)
- [Installation et prerequis](#installation-et-prerequis)
- [CLI : commandes essentielles](#cli--commandes-essentielles)
- [Jobs, profils et batch](#jobs-profils-et-batch)
- [Configuration denrichissement et mode adaptatif](#configuration-denrichissement-et-mode-adaptatif)
- [Sorties et dictionnaires de donnees](#sorties-et-dictionnaires-de-donnees)
- [Persistance budgets et reprise](#persistance-budgets-et-reprise)
- [Observabilite et monitoring](#observabilite-et-monitoring)
- [Tests et qualite](#tests-et-qualite)
- [Contribuer](#contribuer)
- [Support et roadmap](#support-et-roadmap)
- [Limitations connues](#limitations-connues)

## Contexte et objectifs

- Automatiser la generation de profils d'entreprises a forte valeur ajoutee (sites, emails, telephones, LinkedIn, signaux Google Maps).
- Industrialiser la qualite via des budgets (requetes HTTP, octets, temps, RAM) et des KPI formels (`quality_score`, `lines_per_s`, duplication, etc.).
- Permettre des campagnes multi-NAF, reprises automatique (`--resume`) et modes batch resilients pour les equipes Data, Growth et Ops.
- Servir de socle aux orchestrateurs Prefect/Dagster/Airflow (`orchestration/`) et a la stack monitoring Prometheus/Grafana (`monitoring/`).

## Architecture et flux

| Phase | Modules clefs | Sorties principales |
| --- | --- | --- |
| Collecte | `dumps.collect`, `api.collect`, `http.serp`, `crawl.site*`, `headless.collect`, `feeds.collect`, `pdf.collect` | Dumps SIRENE, SERP/ sitemap (`serp/serp_results.parquet`, `crawl/pages.parquet`), pages headless |
| Parsing & normalisation | `parse.*`, `normalize/standardize.py` | `normalized/normalized.parquet`, `reports/standardize_summary.json`, KPI `lines_per_s` |
| Enrichissement | `enrich.*`, `scraper.maps`, `correlation.checks` | `contacts/contacts.parquet`, `domains_enriched.*`, `linkedin_enriched.*`, `metrics/summary.json` |
| Qualite & packaging | `quality.*`, `package/exporter.py`, `finalize.premium_dataset` | `contacts/contacts_clean.parquet`, `dataset.csv`, `dataset_enriched.parquet`, rapport qualite |
| Observabilite | `metrics.export`, `utils/budget_middleware.py`, `metrics/collector.py` | `reports/report_metrics.json`, resume KPI CLI, exposition Prometheus |

Les etapes sont referencees dans `builder_cli.STEP_REGISTRY`, synchronisees via `STEP_DEPENDENCIES` et groupees pour l'execution parallele avec `build_execution_batches`. `utils/directories.initialize_pipeline_directories` cree l'arborescence standard (`normalized/`, `contacts/`, `logs/`, `reports/`, `metrics/`).

## Installation et prerequis

**Prerequis**

- Python 3.11+, `pip`, `virtualenv`.
- `playwright` et navigateurs installes (`python -m playwright install`) pour les etapes headless.
- Acces reseau aux fournisseurs SERP/API et (optionnel) proxy HTTP/HTTPS.
- Docker Compose si vous souhaitez lancer la stack de monitoring.
- Copie de `.env` (depuis `.env.example`) contenant les cles API et secrets requis.

**Linux / macOS (bash/zsh)**

```bash
git clone https://github.com/infinitemoula/projects.git
cd projects
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install
cp .env.example .env  # renseignez vos secrets
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/infinitemoula/projects.git
Set-Location projects
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install
Copy-Item .env.example .env
```

## CLI : commandes essentielles

Toutes les commandes se lancent depuis la racine du repo.

### `run-profile`

```bash
python builder_cli.py run-profile \
  --job jobs/naf_6202A.yaml \
  --input data/sirene_sample.parquet \
  --out out/naf_6202A_quick \
  --profile standard \
  --sample 200 \
  --resume \
  --parallel --workers 6 \
  --prometheus-port 9099 \
  --metrics-file reports/report_metrics.json
```

Options utiles :

| Option | Description |
| --- | --- |
| `--sample N` | Limite temporairement le nombre de lignes traitees. |
| `--resume` | Ignore les etapes dont les sorties (`STEP_OUTPUT_HINTS`) sont deja presentes. |
| `--dry-run` | Valide la config et affiche le DAG sans lancer les etapes. |
| `--json` | Retourne un resume JSON (run_id, KPI, budgets, sorties). |
| `--parallel`, `--parallel-mode {thread,process}`, `--workers N` | Active l'execution parallele des etapes independantes. |
| `--concurrency`, `--per-domain-rps`, `--user-agent-pool` | Surcouche les limites de crawling/enrichissement. |
| `--time-budget-min`, `--max-ram-mb` | Budgets globaux supplementaires. |
| `--use-domains/--no-domains`, `--use-contacts`, `--use-linkedin`, `--use-contacts-ai`, `--use-correlation`, `--use-metrics-export` | Force les flags de `config/enrichment.yaml`. |
| `--prometheus-port`, `--prometheus-address` | Expose les metriques runtime (voir Observabilite). |

Ajoutez `--explain` pour afficher la resolution topologique d'un profil.

### `run-step`

```bash
python builder_cli.py run-step \
  --job jobs/naf_6202A.yaml \
  --step normalize.standardize \
  --input data/sirene_sample.parquet \
  --out out/debug \
  --resume --debug
```

Utile pour re-jouer une etape, inspecter des sorties intermediaires ou deboguer un step en particulier.

### `resume`

`python builder_cli.py resume --job ... --out ... --profile standard` relance automatiquement les etapes manquantes en se basant sur les outputs declares dans `STEP_OUTPUT_HINTS` et sur les etats `SequentialRunState` (crawl/ SERP/ headless).

### `batch`

```bash
python builder_cli.py batch \
  --naf 6202A --naf 6920Z \
  --template job_template.yaml \
  --profile standard_nocapital_v2 \
  --input data/sirene_sample.parquet \
  --output-dir out/multi_naf \
  --jobs-dir jobs_generated \
  --continue-on-error \
  --json
```

Le mode batch genere un job YAML par NAF (`create_job.generate_niche_name`) dans `jobs_generated/`, applique les flags passes en ligne de commande et cree un sous-dossier `out/multi_naf/<naf_code>/` par execution. Les resultats resumeent les succes, echecs, durees et KPI.

## Jobs, profils et batch

Extrait simplifie de [`job_template.yaml`](job_template.yaml) :

```yaml
niche: "{niche_name}"
filters:
  naf_include: ["{naf_code}"]
profile: "{profile}"
output:
  dir: "out/{niche_name}"
kpi_targets:
  min_quality_score: 80
  min_lines_per_s: 50
budgets:
  max_http_requests: 2000
  max_http_bytes: 52_428_800
  time_budget_min: 90
scraper:
  maps:
    per_host_rps: 0.4
    proxy:
      enabled: true
      use_env: true
```

Points clefs :

- `filters` : restreignent les entreprises (NAF, regions, statut) avant normalisation.
- `profile` : selectionne la liste d'etapes `PROFILES[...]`. `steps_order` peut forcer un ordre custom.
- `kpi_targets`: alimente `utils.budget_middleware.KPICalculator`. Les alertes apparaissent en fin de run.
- `budgets`: limites consommees par `BudgetTracker` (HTTP requests/bytes, temps).
- `scraper.*` : throttling Google Maps/Playwright, proxies et retention.

Profils disponibles (voir `builder_cli.PROFILES`) :

| Profil | Contenu |
| --- | --- |
| `quick` | Collecte dumps/API, normalisation, controle qualite et export premium rapide. |
| `standard` | Ajoute feeds JSON-LD, enrichissements adresse/Google Maps/domaine/email/phone et LinkedIn Clearbit. |
| `standard_nocapital` | Variante zero capital : mise sur `enrich.domains`, `enrich.contacts`, `correlation.checks`, `enrich.linkedin`. |
| `standard_nocapital_v2` | Meme base que ci-dessus + export metriques et overrides (`PROFILE_FEATURE_OVERRIDES`). |
| `hybrid` | Pipeline centre sur SERP + crawl async + nettoyage contacts (initial/final) + fallback headless. |
| `deep` | Profil exhaustif : headless, feeds, PDF, parsing multi formats, enrichissements complets. |
| `internal` | Profil interne incluant `scraper.maps` et `quality.enrich_score` pour score marketing. |

Le sous-commande `batch` combine ces profils avec `create_job.py` pour generer des jobs par NAF ou niche.

## Configuration denrichissement et mode adaptatif

`config/enrichment.yaml` est valide par `config/enrichment_config.py` (Pydantic). Sections importantes :

- **Flags globaux** : `use_domains`, `use_contacts`, `use_linkedin`, `use_correlation`, `use_metrics_export`, `ai.enabled`. Les options CLI `--use-*` les surchargent.
- **Cache & circuit breaker** : configure `.cache/http.sqlite`, TTL, politiques de retry communes aux modules.
- **`domains`** : fournisseurs SERP (`providers_config`), HTTP client (`max_concurrent_requests`, `per_host_limit`, `user_agents_file`, `cache_dir`), heuristiques TLD/prefixes, embeddings.
- **`contacts`** : pages priorisees, limites sitemap, respect robots, listes d'emails/ prefixes generiques.
- **`linkedin`** : meme schema que `domains` pour trouver des pages `linkedin.com/company`.
- **`ai`** : active l'extraction contacts basee sur IA si `contacts: true`.
- **`adaptive`** : pilote `utils.adaptive_controller.AdaptiveController`. Quand `enabled: true`, le controller ajuste `chunk_size` et `max_concurrent_requests` a partir du taux d'erreur, RAM et debit observe (voir `_apply_adaptive_feedback` dans `builder_cli.py`).

Les budgets peuvent aussi etre charges via `config/budget_config.py` (fichier `config/budgets.yaml` ou variables `BUDGET_*`). `kpi_targets` + budgets alimentent `BudgetTracker` et `KPICalculator`, exposes dans les logs JSON et dans le resume CLI.

## Sorties et dictionnaires de donnees

Chaque run cree un `out/<niche>/` (exemple : `out/naf_6202A_quick/`) contenant :

| Chemin | Description |
| --- | --- |
| `dataset.csv`, `dataset.parquet` | Jeu standardise (post `package.export`). |
| `dataset_enriched.*` | Variante incluant les contacts et metadonnees additionnelles. |
| `normalized/normalized.{csv,parquet}` | Base SIRENE nettoyee et aggregee. |
| `contacts/contacts.parquet`, `contacts/contacts_clean.parquet`, `contacts/no_contact.csv` | Contacts bruts vs nettoyes, stats de rejet. |
| `logs/<run_id>.json` | Journal structure (un JSON par etape) cree via `utils.io.log_json`. |
| `reports/standardize_summary.json` | KPI de normalisation (`lines_per_s`, champs obligatoires, duplications). |
| `reports/report_metrics.json` | Export du collecteur `metrics/collector.py` (counters, latences, RAM). |
| `metrics/summary.json`, `metrics/scraper_stats.csv` | Metriques specifiques a `scraper.maps` et a `metrics.export`. |
| `quality_checks.json`, `quality_score.parquet`, `quality_summary.json` | Statistiques de qualite contact/export. |
| `manifest.json`, `sha256.txt`, `data_dictionary.md` | Metadonnees exploitables par les equipes marketing/ops. |

Le dictionnaire de donnees contacts est documente dans `docs/data_dictionary_contacts.md`; les exports principaux incluent aussi `out/.../data_dictionary.md`.

## Persistance budgets et reprise

- `build_context` genere un `run_id`, cree les dossiers (`utils.directories`), charge l'environnement (`config.load_env`) et garde la trace des sorties recemment produites pour eviter les doublons en mode `--resume`.
- `STEP_OUTPUT_HINTS` indique quelles sorties verifiees avant de marquer un step `SKIPPED`. Les etats de crawl/serp/headless sont stockes dans `crawl_state.json`, `serp_state.json`, etc., ce qui permet de reprendre un crawl interrompu.
- `BudgetTracker` surveille requetes HTTP, octets, temps et RAM (cf. `--max-ram-mb`). En cas de depassement, les etapes echouent avec `BUDGET_EXCEEDED` et le resume CLI inclut la cause.
- `KPICalculator` consolide `kpi_targets` et ajoute un suffixe `(KPIs MET|NOT MET)` au message final.
- `metrics.collector.Metrics` imprime une ligne `KPI SUMMARY | ...` et peut ecrire un JSON custom via `--metrics-file`.

## Observabilite et monitoring

- `--prometheus-port` active `monitoring/prometheus_exporter.py` qui expose `pipeline_step_duration_seconds`, `pipeline_step_status_total`, `pipeline_run_info`, etc. (voir `docs/observability.md`).
- `monitoring/docker-compose.yml` fournit un exemple cle-en-main : `pipeline-worker`, `prometheus` et `grafana` pre-configures (`monitoring/grafana/provisioning/...`).
- `metrics.export` ecrit `reports/report_metrics.json` et `metrics/summary.json`; vous pouvez les pousser vers vos observabilites custom.
- `docs/observability.md` et `docs/orchestration.md` detaille l'integration a Grafana, Prefect, Dagster et Airflow.

## Tests et qualite

- **Tests unitaires / integratifs** : `pytest` (ou `pytest tests/test_builder_cli.py -k resume` pour cibler le resume). La suite couvre budgets, collecte, enrichissements (Google Maps, domains, contacts IA), export et CLI.
- **Lint / format** : `ruff check .` et `ruff format` (config `pyproject.toml`, ligne max 100). `black` reste compatible avec cette configuration.
- **Playwright** : si vous modifiez `headless.collect*`, executez les tests associes (`tests/test_internal_crawler.py`, `tests/test_maps_scraper.py`).
- **Couverture** : badge base sur `.github/badges/coverage.json`, mis a jour par la CI GitHub Actions (`ci.yml`).

Pensez a ajouter des tests lorsque vous introduisez une nouvelle etape ou modifiez un format d'output.

## Contribuer

1. Ouvrez une issue decrivant bug/feature, avec commande lancee, extraits de logs (`out/.../logs/*.json`) et echantillon input si possible.
2. Creez une branche (`feature/...`, `fix/...`), faites tourner `ruff` et `pytest`.
3. Mettez a jour la documentation (README, `docs/`, jobs/profils) et indiquez les impacts budgets/exports dans la PR.
4. Ne committez jamais de donnees sensibles (contenu `out/`, `models/`, caches) ni de secrets `.env`.

## Support et roadmap

- Support interne : canal Slack `#b2b-data-pipeline` ou `data@infinitemoula.com`.
- Support externe : issues GitHub avec details (commande, version Python, profil, extraits de log/KPI).
- Documentation complementaire : `docs/enrichment.md`, `docs/orchestration.md`, `docs/observability.md`, `docs/scraping-methods-guide.md`, `QUALITY_FRAMEWORK.md`.

Feuille de route (indicative) :

1. Industrialisation Prefect/Dagster/Airflow (parametrisation automatique des etapes via `STEP_DEPENDENCIES`).
2. Exposition temps reel des budgets (`utils.cost_manager`) et alertes dynamiques.
3. Nouveaux connecteurs API (Clearbit complet, Pappers, reseaux sociaux) avec budgets dedies.
4. Observabilite headless (latence Playwright, erreurs reseau) et etiquettes par domaine dans Prometheus.

## Limitations connues

- **Artefacts ML** : les modeles pickles (`ml/address_classifier.py`, `ml/extraction_models.py`, `utils/ml_optimizer.py`) supposent la meme version de scikit-learn entre entrainement et inference.
- **KPI `lines_per_s`** : l'indicateur depend des steps qui exposent `total_lines_processed`. Si une etape intermediaire saute cette metrique, le KPI tombe a 0 et peut declencher une alerte.
- **Preconditions sorties** : certains modules supposent l'existence de fichiers (`normalized.parquet`, dossier `contacts/`). Gardez `utils.directories.initialize_pipeline_directories` et `STEP_OUTPUT_HINTS` a jour si vous ajoutez des etapes.
- **Metriques headless** : les etapes Playwright n'exportent pas encore leurs latences/erreurs dans Prometheus.
- **Budget d'alerting** : les seuils journaliers de la stack monitoring sont encore configures a la main; un branchement direct sur `config/budget_config.py` reste a faire.

Bonne exploration du pipeline !
