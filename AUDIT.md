# Audit technique – Pipeline de données B2B françaises

_Date_: 2025-11-10  
_Portée_: revue du dépôt `projects/` couvrant la CLI `builder_cli.py`, les profils/pipelines, la configuration d’enrichissement, l’observabilité et la qualité logicielle.

## 1. Synthèse exécutive

- **Forces**
  - Orchestration unifiée via `builder_cli.py` avec registre d’étapes, dépendances et reprise automatique (`--resume`) couvrant l’ensemble du cycle collecte → enrichissement → packaging (`builder_cli.py:109–270`).
  - Gouvernance budgétaire et KPI intégrée (`utils/budget_middleware.py`, `config/budget_config.py`) permettant de borner coûts réseau, temps et RAM, et de signaler les dérives qualité.
  - Documentation à jour (README, `docs/*.md`) et gabarits de jobs prêt à l’emploi (`job_template.yaml`) qui cadrent les équipes Ops/Growth.
  - Suite de tests Pytest fournie (`tests/`), couplée à la CI GitHub Actions avec badge de couverture.
  - Observabilité optionnelle via Prometheus/Grafana (`monitoring/`) et export JSON (`metrics/exporter`).

- **Risques / dettes**
  - Pas de versionnement explicite des artefacts ML (risque d’incompatibilité scikit-learn).
  - Dépendance forte aux outputs intermédiaires (si un fichier manque, l’étape échoue sans remède automatique).
  - Budget d’alerting partiellement codé en dur côté monitoring.
  - Absence de métriques détaillées pour les étapes headless/Playwright.

## 2. Architecture fonctionnelle

| Phase | Étapes représentatives | Fichiers de référence |
| --- | --- | --- |
| Collecte | `dumps.collect`, `api.collect`, `http.serp`, `crawl.site*`, `headless.collect*` | `builder_cli.py:64` (STEP_DEPENDENCIES), `nethttp/*`, `headless/*` |
| Parsing & normalisation | `parse.html/jsonld/pdf`, `normalize.standardize` | `builder_cli.py:160`, `normalize/standardize.py` |
| Enrichissement | `enrich.address/domain/site/email/phone`, `enrich.linkedin*`, `scraper.maps`, `correlation.checks` | `builder_cli.py:151`, `enrich/*`, `scraper/maps_scraper.py` |
| Qualité & packaging | `quality.*`, `package.export`, `finalize.premium_dataset` | `builder_cli.py:162`, `package/exporter.py` |
| Observabilité | `metrics.export`, Prometheus exporter | `builder_cli.py:151`, `monitoring/prometheus_exporter.py` |

Les profils (`quick`, `standard`, `standard_nocapital[_v2]`, `hybrid`, `deep`, `internal`) sont définis dans `builder_cli.py:160–265`. Le tri topologique et la parallélisation s’appuient sur `pipeline.topo_sort` et `build_execution_batches` (`builder_cli.py:830–870`).

## 3. CLI & orchestration

- `run-profile`, `run-step`, `resume`, `batch` définis dans `main()` (`builder_cli.py:1465–1675`) avec sous-parseurs dédiés.
- Options majeures : budgets (`--time-budget-min`, `--max-ram-mb`), parallélisme (`--parallel`, `--workers`, `--parallel-mode`), override d’enrichissements (`--use-domains`, etc.), exposition Prometheus (`--prometheus-port`, `--prometheus-address`), export métriques (`--metrics-file`).
- Batch multi-NAF : `run_batch_jobs` et `prepare_multi_naf_runs` (`builder_cli.py:897–1096`) orchestrent la génération de jobs (via `create_job`) et l’exécution séquentielle avec tolérance aux erreurs (`--continue-on-error`).
- Orchestrations externes : `docs/orchestration.md`, `orchestration/{prefect_flow.py,dagster_job.py,airflow_dag.py}` réutilisent le même registre pour éviter la divergence.

## 4. Jobs & configuration

- `job_template.yaml` fournit un gabarit incluant filtres NAF, budgets, KPI et throttling Google Maps.
- `create_job.py` gère la génération automatisée (CLI et intégration pipeline).
- `config/enrichment.yaml` (valide par `config/enrichment_config.py`) centralise les réglages SERP/crawling : pool de User-Agent, caches (`.cache/http.sqlite`), circuit breaker, heuristiques TLD/prefixes, modules AI, configuration adaptive.
- `config/budget_config.py` permet de surcharger les seuils via YAML ou variables d’environnement `BUDGET_*`.
- Overrides par profil : `PROFILE_FEATURE_OVERRIDES` (`builder_cli.py:270–293`) appliquent des flags/config spécifiques (ex. `standard_nocapital_v2` active `use_metrics_export`).

## 5. Données & outputs

Structure typique d’un run sous `out/<niche>/` :

- `dataset.{csv,parquet}`, `dataset_enriched.{csv,parquet}` : export final (`package/exporter.py:1–210`).
- `normalized/normalized.{csv,parquet}`, `reports/standardize_summary.json`.
- `contacts/contacts.parquet`, `contacts/contacts_clean.parquet`, `contacts/no_contact.csv`.
- `logs/<run_id>.json` (journal structuré append-only, `utils/io.py:55–89`).
- `reports/report_metrics.json`, `metrics/summary.json`, `metrics/scraper_stats.csv`.
- `quality_checks.json`, `quality_score.parquet`, `quality_summary.json`.
- `manifest.json`, `data_dictionary.md`, `sha256.txt` pour la traçabilité.

La reprise (`--resume`) s’appuie sur `STEP_OUTPUT_HINTS` (`builder_cli.py:296–332`) et sur les fichiers d’état séquentiel (`crawl_state.json`, `serp_state.json`, etc.).

## 6. Budgets, KPI et adaptativité

- `build_context` initialise `BudgetTracker`/`KPICalculator` via `budget_middleware.create_*` (`builder_cli.py:689–706`) et applique les budgets CLI/Job.
- `BudgetTracker` surveille requêtes HTTP, octets, temps et RAM (`builder_cli.py:423–520` effectue les checks avant/après chaque étape ; `_check_ram` à `builder_cli.py:384–391`).
- KPI finaux : `kpi_calculator.calculate_final_kpis` logue et enregistre les résultats (`builder_cli.py:1265–1305`), suffixant le message final avec `KPIs MET/NOT MET`.
- Metrics collector : `metrics/collector.py` accumule compteurs, latences et ressources, exporte JSON et imprime `KPI SUMMARY | ...` (`metrics/collector.py:213–229`).
- Mode adaptatif : `AdaptiveController` (`utils/adaptive_controller.py`) ajuste `chunk_size`/`max_concurrent_requests` selon erreur, débit et RAM. Hooks dans `_apply_adaptive_feedback`/`_update_adaptive_context` (`builder_cli.py:565–625`).

## 7. Observabilité et monitoring

- Export Prometheus activable via `--prometheus-port`; `monitoring/prometheus_exporter.py` expose histogrammes et compteurs par étape.
- Stack Docker Compose (`monitoring/docker-compose.yml`) fournit Prometheus + Grafana + worker d’exemple (voir `docs/observability.md`).
- Alerting suggéré (pannes d’étape, latence P95, absence de données).
- Limite actuelle : pas de métriques détaillées pour Playwright/headless (risque identifié section 10).

## 8. Qualité logicielle & tests

- `pyproject.toml` : formatage Black (100 caractères), Ruff (E,F,B,I,UP), mypy partiel.
- `requirements.txt` couvre pandas/pyarrow/httpx/playwright/bs4/lxml… + dépendances ML (numpy, sklearn, joblib).
- Suite Pytest fournie (`tests/`), avec couverture des domaines suivants : budgets (`tests/test_budget_middleware.py`), CLI (`tests/test_builder_cli.py`, `tests/test_batch_jobs.py`), enrichissements (`tests/test_enrich_*`, `tests/test_google_maps_search.py`), pipeline complet (`tests/test_pipeline.py`), export (`tests/test_exporter_*`), modules ML/AI (`tests/test_adaptive.py`, `tests/test_ai_fallback.py`).
- CI GitHub Actions (badge `ci.yml`) + badge de couverture (`.github/badges/coverage.json`).

## 9. Documentation

- README (mis à jour) couvre architecture, commandes, profils, configuration, observabilité et limitations.
- `docs/` contient des guides dédiés : enrichissement (`docs/enrichment.md`), orchestration, observabilité, scraping, dictionnaire contacts, aspects légaux, notes d’upgrade.
- `QUALITY_FRAMEWORK.md` détaille les contrôles qualité.

## 10. Risques et dettes prioritaires

1. **Versionnement artefacts ML**  
   - _Impact_: incompatibilités silencieuses lors des upgrades scikit-learn.  
   - _Fichiers_: `ml/address_classifier.py`, `ml/extraction_models.py`, `utils/ml_optimizer.py`.  
   - _Action_: stocker hash/version de pipeline dans les métadonnées, prévoir migration automatique.

2. **Dépendance aux outputs physiques**  
   - _Impact_: un run peut échouer si un fichier intermédiaire est manquant/corrompu malgré `--resume`.  
   - _Action_: introduire des checksums/contrats d’outputs et un mécanisme de re-génération ciblée.

3. **Budgets monitoring codés en dur**  
   - _Impact_: seuils non synchronisés entre CLI et stack Prometheus, risque d’alertes non pertinentes.  
   - _Action_: externaliser les seuils dans `config/budget_config.py` et les injecter côté dashboard/alertes.

4. **Observabilité Playwright limitée**  
   - _Impact_: difficile de diagnostiquer les coûts/erreurs headless (`headless.collect*`).  
   - _Action_: instrumenter Playwright (durée navigation, nombre de pages, codes HTTP) et exposer via Prometheus.

5. **Gestion secrets / .env**  
   - _Impact_: `.env` non documenté côté sécurité; valider la non-commitation mais prévoir un vault centralisé.  
   - _Action_: compléter la doc sécurité (chiffrement, rotation des clés).

## 11. Recommandations

1. **Mettre en place le versionnement des modèles ML** (haute priorité).  
   Ajouter un champ `model_version` et `sklearn_version` dans les artefacts, refuser/retourner une erreur claire en cas de mismatch.

2. **Centraliser les budgets/alertes** (haute priorité).  
   Créer un fichier `monitoring/budgets.yaml` consommé par la CLI et Grafana pour assurer la cohérence.

3. **Instrumenter Playwright/headless** (moyenne priorité).  
   Ajouter des hooks dans `headless.collect` pour reporter temps de chargement, taille des pages et codes d’erreur au collector Prometheus.

4. **Automatiser la vérification des outputs** (moyenne priorité).  
   Implémenter un validateur post-step qui vérifie existence + schéma minimal avant de marquer une étape `OK`.

5. **Documenter un run de référence** (basse priorité).  
   Capturer un « blueprint » (`docs/run_reference.md`) décrivant un run `standard` complet avec traces et KPI pour faciliter l’onboarding.

## 12. Annexes

- Code source principal : `builder_cli.py`, `utils/*`, `enrich/*`, `normalize/*`, `quality/*`.
- Configuration : `job_template.yaml`, `config/enrichment.yaml`, `config/budget_config.py`.
- Observabilité : `monitoring/*`, `docs/observability.md`.
- Qualité documentaire : `README.md`, `docs/*.md`, `QUALITY_FRAMEWORK.md`.
- Tests : dossier `tests/`.

## 13. Cartographie des fonctionnalités par fichier

### Cœur CLI & orchestration
- `builder_cli.py` : registre d’étapes (`STEP_REGISTRY`), dépendances (`STEP_DEPENDENCIES`), profils (`PROFILES`), parsing CLI (`run-profile`, `run-step`, `resume`, `batch`), gestion budgets/KPI/Prometheus, parallélisation (`build_execution_batches`), reprise (`STEP_OUTPUT_HINTS`).
- `utils/pipeline.py` : configuration logger, résolution dynamique des étapes (`resolve_target`), tri topologique (`topo_sort`), journalisation structurée (`log_step_event`).
- `create_job.py` + `job_template.yaml` : génération de jobs YAML par NAF, substitution de paramètres, CLI autonome pour produire des fichiers `jobs/*.yaml`.
- `orchestration/prefect_flow.py`, `orchestration/dagster_job.py`, `orchestration/airflow_dag.py` : wrappers pour lancer les étapes CLI dans Prefect, Dagster ou Airflow en important le même registre.

### Utilitaires & configuration
- `utils/budget_middleware.py` : `BudgetTracker`, `KPICalculator`, enforcement HTTP/time/RAM, calcul des KPI `quality_score`, `lines_per_s`, duplication, etc.
- `utils/adaptive_controller.py` : heuristiques d’ajustement de `max_concurrent_requests` et `chunk_size` selon erreurs, débit, RAM.
- `utils/io.py` : I/O sûrs (écritures atomiques, CSV, JSON, hashing, logs JSON).
- `utils/directories.py` : création standard de l’arborescence `out/<run>/` (`normalized/`, `contacts/`, `logs/`, `reports/`, `metrics/`).
- `utils/config.py` (et sous-modules) : chargement `.env`, validation des variables requises.
- `config/enrichment_config.py` : schéma Pydantic pour `config/enrichment.yaml` (HTTP clients, SERP providers, heuristiques, AI/adaptive).
- `config/budget_config.py` : lecture des seuils (`BudgetThresholds`) via YAML + env, exposée par `get_budget_thresholds`.
- `config/enrichment.yaml` : configuration par défaut (SERP providers, cache, user-agents, flags `use_*`, mode adaptatif).

### Collecte & parsing
- `dumps/*.py` : ingestion des exports SIRENE et autres dumps.
- `nethttp/*` : collecte HTTP statique, SERP, sitemap, crawling synchrone/asynchrone, gestion d’état `SequentialRunState`.
- `headless/*` : collecte Playwright (pages dynamiques) + fallback.
- `feeds/*`, `pdf/*`, `api/*` : collecte de flux RSS/JSON et documents PDF/API externes.
- `parse/*.py` : extraction structurée depuis HTML/JSON-LD/PDF, pipeline contacts (incluant IA).
- `normalize/standardize.py` : nettoyage SIRENE, calcul KPI `lines_per_s`, exports `standardize_summary.json`.

### Enrichissement
- `enrich/address_search.py` : recherche d’adresses, géocodage, préparation Google Maps.
- `enrich/google_maps_search.py` : requêtes Maps avec budgets et heuristiques.
- `enrich/domain_discovery.py`, `enrich/enrich_domains.py` : découverte de domaines via SERP/heuristiques.
- `enrich/site_probe.py`, `enrich/dns_checks.py` : validation sites/DNS.
- `enrich/email_heuristics.py`, `enrich/phone_checks.py` : extraction/score d’emails et téléphones.
- `enrich/enrich_contacts.py`, `enrich/enrich_linkedin.py`, `enrich/enrich_linkedin_clearbit_lite.py` : consolidation des contacts, recherche LinkedIn/équivalent Clearbit.
- `scraper/maps_scraper.py` : scraping ciblé Google Maps avec throttling configurable.
- `correlation/correlation.py` : vérifications croisées (domaines ↔ entités).

### Qualité & packaging
- `quality/checks.py`, `quality/score.py`, `quality/enrich_score.py`, `quality/dedupe.py`, `quality/clean_contacts*.py` : pipeline qualité (contrôles, scoring, déduplication, nettoyage contacts).
- `quality/validation.py` : validations champ par champ (emails, sites, téléphones, LinkedIn).
- `package/exporter.py` : fusion dataset normalisé + contacts, calcul stats de validation, préparation `dataset_enriched.*`.
- `utils/filters.py` (`run_finalize_premium_dataset`) : filtrage final pour le dataset premium.

### Observabilité & métriques
- `metrics/collector.py` : agrégation en mémoire (counters, latences, ressources), export JSON (`reports/report_metrics.json`), résumé CLI.
- `metrics/exporter.py` : écriture dédiée des métriques pipeline.
- `monitoring/prometheus_exporter.py` : serveur Prometheus (histogrammes par étape, run info).
- `monitoring/docker-compose.yml` + `monitoring/grafana/*` : stack Prometheus/Grafana clé en main.
- `docs/observability.md` : procédures d’exploitation, alertes recommandées.

### Documentation & conformité
- `README.md` : vision d’ensemble (architecture, commandes, profils, config, observabilité, limitations).
- `docs/*.md` : guides spécialisés (enrichissement, orchestration, scraping, dictionnaire contacts, légalité, upgrade notes).
- `QUALITY_FRAMEWORK.md` : cadre de contrôle qualité.
- `ENRICHMENT_IMPLEMENTATION.md` : détails techniques sur les modules d’enrichissement.

### Tests & validation
- `tests/` : suite Pytest couvrant budgets, CLI, enrichissements (SERP, contacts, LinkedIn), parsing, qualité, exporter, adaptatif, AI fallback, etc.
- `.github/workflows/ci.yml` : pipeline CI (lint/tests + mise à jour badge couverture).

_Fin de l’audit._
