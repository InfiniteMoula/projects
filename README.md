# Pipeline de données B2B françaises

[![CI](https://github.com/infinitemoula/projects/actions/workflows/ci.yml/badge.svg)](https://github.com/infinitemoula/projects/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/infinitemoula/projects/main/.github/badges/coverage.json)](https://github.com/infinitemoula/projects)

> Ce README s'appuie sur la checklist de [makeareadme.com](https://www.makeareadme.com/) pour couvrir toutes les informations utiles du projet.

## Sommaire
- [Contexte et objectifs](#contexte-et-objectifs)
- [Fonctionnalités clés](#fonctionnalités-clés)
- [Architecture et organisation du code](#architecture-et-organisation-du-code)
- [Installation et dépendances](#installation-et-dépendances)
- [Utilisation : commandes essentielles](#utilisation--commandes-essentielles)
- [Configuration des jobs et des profils](#configuration-des-jobs-et-des-profils)
- [Persistance et reprise](#persistance-et-reprise)
- [Tests et qualité](#tests-et-qualité)
- [Limitations connues et axes d'amélioration](#limitations-connues-et-axes-damélioration)
- [Contribuer](#contribuer)
- [Licence et crédits](#licence-et-crédits)
- [Badges et visuels suggérés](#badges-et-visuels-suggérés)
- [Support et roadmap](#support-et-roadmap)

## Contexte et objectifs

Ce dépôt regroupe le pipeline industriel utilisé par l'équipe Infinitemoula pour collecter, enrichir et qualifier des données B2B françaises (principalement issues de la base SIRENE). L'outil répond aux besoins suivants :

- Automatiser la création de profils d'entreprises à forte valeur ajoutée (contacts, dirigeants, données web).
- Assurer un suivi qualité et budgétaire fin afin de fiabiliser la production de datasets premium.
- Servir de socle commun aux équipes Data, Growth et Ops pour orchestrer des campagnes multi-NAF à grande échelle.

Les objectifs fonctionnels sont de proposer des profils d'exécution adaptés (rapide, standard, profond), des enrichissements multi-sources, un reporting détaillé et une reprise facile des traitements. Côté business, le pipeline vise à accélérer la génération de leads qualifiés et à maîtriser les coûts d'acquisition (HTTP, API, Playwright) grâce aux budgets configurables.

## Fonctionnalités clés

- CLI unifié [`builder_cli.py`](builder_cli.py) exposant `run-profile`, `run-step`, `resume` et `batch`, avec support Prometheus et JSON.
- Profils d'orchestration prêts à l'emploi (`quick`, `standard`, `hybrid`, `deep`, `internal`) couvrant collecte, parsing, enrichissement et packaging.
- Enrichissements spécialisés : Google Maps, découverte de domaines, vérification DNS, heuristiques emails/téléphones, crawling synchrone/asynchrone.
- Normalisation SIRENE et génération de rapports KPI (`normalize/standardize.py`) avec export CSV/Parquet consolidé.
- Suivi budgétaire et KPI configurable (HTTP, temps, RAM, quality score, lines_per_s) via [`utils/budget_middleware.py`](utils/budget_middleware.py).
- Modèles ML scikit-learn pour qualifier des adresses et optimiser les paramètres (`ml/address_classifier.py`, `utils/ml_optimizer.py`).
- Monitoring et observabilité optionnels (export Prometheus, stack Grafana, alerting) dans [`monitoring/`](monitoring).
- Suite de tests riche couvrant middleware, pipelines, enrichment et qualité dans [`tests/`](tests).

## Architecture et organisation du code

| Composant | Description |
| --- | --- |
| [`builder_cli.py`](builder_cli.py) | Point d'entrée CLI, résout les étapes via `STEP_REGISTRY`, gère les profils, la parallélisation, les budgets et l'export Prometheus. |
| [`utils/`](utils) | Couche utilitaire : budgets et KPIs (`budget_middleware.py`), état séquentiel (`state.py`), I/O atomiques, configuration dynamique, suivi des coûts et métriques. |
| [`ml/address_classifier.py`](ml/address_classifier.py) & [`ml/extraction_models.py`](ml/extraction_models.py) | Modèles scikit-learn pour noter la qualité des adresses et apprendre des motifs d'extraction, persistant leurs artefacts dans `models/`. |
| [`normalize/standardize.py`](normalize/standardize.py) | Standardisation des dumps SIRENE, calcul des KPI `lines_per_s`, génération du rapport `reports/standardize_summary.json`. |
| [`nethttp/`](nethttp) | Collecte HTTP/ SERP / sitemap et crawling (sync & async) avec persistance `SequentialRunState` (`crawl_state.json`, `serp_state.json`). |
| [`enrich/`](enrich) | Modules d'enrichissement (Google Maps, discovery de domaines, heuristiques emails/téléphones, site probing). |
| [`parse/`](parse) | Extraction structurée depuis HTML, PDF, JSON-LD et pipeline de contacts (dont version IA). |
| [`quality/`](quality) | Contrôles qualité, nettoyage et scoring enrichi des contacts. |
| [`package/exporter.py`](package/exporter.py) | Fusion des données normalisées et des contacts pour créer `dataset_enriched.csv` / `.parquet`. |
| [`jobs/`](jobs) | Jobs YAML prêts à l'emploi (gabarits, budgets, KPI), avec un modèle générique dans [`job_template.yaml`](job_template.yaml). |
| [`tests/`](tests) | Suite Pytest couvrant CLI, middleware budget, parsing, enrichissements, pipeline complet. |
| [`monitoring/`](monitoring) | Exporter Prometheus, alert manager et stack Grafana (voir `docker-compose.yml`). |
| [`orchestration/`](orchestration) | Intégrations Airflow, Dagster et Prefect pour piloter le pipeline dans des orchestrateurs externes. |
| [`docs/`](docs) | Documentation complémentaire : guides scraping, orchestrations, observabilité et considérations légales. |
| [`data/`](data) | Jeux d'essai (ex. `sirene_sample.parquet`) utiles pour les profils de test. |

## Installation et dépendances

### Prérequis

- Python 3.11 et `pip`.
- Accès Internet pour les appels HTTP, APIs et Playwright.
- Optionnel : Docker si vous souhaitez lancer la stack monitoring (`monitoring/docker-compose.yml`).
- Clés API pour les services tiers (à renseigner dans `.env` selon vos besoins).

### Mise en place (Unix/macOS)

```bash
git clone https://github.com/infinitemoula/projects.git
cd projects
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install  # requis pour les étapes headless
cp .env.example .env          # complétez les clés nécessaires
```

### Mise en place (Windows PowerShell)

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

> Astuce : vous pouvez limiter l'installation aux dépendances critiques (`pandas`, `pyarrow`, `httpx`, `psutil`, etc.) pour les usages offline, mais les profils `standard` et `deep` nécessitent l'ensemble de `requirements.txt`.

## Utilisation : commandes essentielles

Toutes les commandes se lancent depuis la racine du dépôt avec l'environnement virtuel activé.

#### Lancer un profil d'enrichissement

```bash
python builder_cli.py run-profile \
  --job jobs/naf_6202A.yaml \
  --input data/sirene_sample.parquet \
  --out out/naf_6202A_quick \
  --profile quick \
  --sample 50 \
  --debug
```

Résultats attendus :

- Normalisation : `out/naf_6202A_quick/normalized.parquet` et rapport `reports/standardize_summary.json`.
- Nettoyage contacts : `out/naf_6202A_quick/contacts/contacts_clean.parquet`.
- Export final : `out/naf_6202A_quick/dataset_enriched.csv` et `.parquet`.
- Journaux structurés : `out/naf_6202A_quick/logs/<run_id>.json`.

#### Visualiser les étapes d'un profil

```bash
python builder_cli.py run-profile \
  --job jobs/naf_6202A.yaml \
  --profile standard \
  --explain
```

La commande affiche la résolution topologique des étapes définies dans `PROFILES`.

#### Reprendre ou cibler une étape

```bash
python builder_cli.py run-step \
  --job jobs/naf_6202A.yaml \
  --step normalize.standardize \
  --input data/sirene_sample.parquet \
  --out out/debug_run \
  --resume \
  --verbose
```

- `--resume` réutilise les sorties déjà présentes (via `SequentialRunState` et les logs JSON).
- `--dry-run` exécute la validation sans lancer les étapes.
- `--json` renvoie la synthèse au format JSON sur stdout.

#### Traitement multi-NAF automatisé

```bash
python builder_cli.py batch \
  --naf 6202A --naf 6920Z \
  --template job_template.yaml \
  --profile standard \
  --input data/sirene_sample.parquet \
  --output-dir out/pro_services \
  --continue-on-error \
  --json
```

La commande génère des jobs spécifiques dans `jobs_generated/`, exécute chaque NAF dans son sous-dossier (`out/pro_services/<naf>/`) et produit un récapitulatif (succès/échecs, KPI, chemins de sortie).

## Configuration des jobs et des profils

Les jobs se décrivent en YAML. Le fichier [`job_template.yaml`](job_template.yaml) fournit une base annotée. Points clés :

```yaml
niche: "experts_comptables"
filters:
  naf_include: ["6920Z"]
profile: "standard"          # surchargé par --profile si besoin
output:
  dir: "out/experts_comptables"
kpi_targets:
  min_quality_score: 80
  min_lines_per_s: 50
budgets:
  max_http_requests: 2000
  max_http_bytes: 52428800
  time_budget_min: 90
scraper:
  maps:
    per_host_rps: 0.4
    delay_range: [2.0, 5.0]
    proxy:
      enabled: true
      use_env: true
```

- **Filtres** : restreignent les entreprises par NAF, activité ou zone (`filters`).
- **Étapes** : la clé `profile` sélectionne un profil défini dans `PROFILES`. Vous pouvez personnaliser l'ordre via `steps_order`.
- **Budgets** : consommés par [`BudgetTracker`](utils/budget_middleware.py) (HTTP, octets, temps, RAM). Dépassement ⇒ `BudgetExceededError`.
- **KPI** : 6 indicateurs (`quality_score`, `dup_pct`, etc.) évalués en fin de run par `KPICalculator`.
- **Scraper** : throttling Google Maps et configuration proxy via variables d'environnement.
- **Export** : `package/exporter.py` fusionne `normalized.parquet` et les contacts.
- **Retention/logs** : `retention_days` peut être exploité par vos scripts de housekeeping.

Des profils adaptés sont déclarés dans [`builder_cli.py`](builder_cli.py) :
- `quick` : collecte + normalisation + scoring.
- `standard` : ajoute feeds, enrichissements et Google Maps.
- `hybrid` : pipeline centré sur le scraping/contacts avec fallback headless.
- `deep` : pipeline complet avec PDF, parsing avancé, enrichissements multiples.
- `internal` : profil utilisé en interne (inclut `scraper.maps` et `quality.enrich_score`).

Référez-vous aux documents dans [`docs/`](docs) pour les bonnes pratiques d'orchestration (`docs/orchestration.md`) et de scraping (`docs/scraping-methods-guide.md`).

## Persistance et reprise

- **État séquentiel** : les collecteurs HTTP et crawling utilisent [`SequentialRunState`](utils/state.py) pour garder la trace des domaines traités (`crawl_state.json`, `serp_state.json`, `google_maps_state.json`). Le flag `--resume` se base sur ces fichiers pour éviter de retraiter les éléments réussis.
- **Logs structurés** : chaque exécution écrit un log JSON (`out/<run>/logs/<run_id>.json`) via `utils.io.log_json`. Ils contiennent status, durée, erreurs et métriques budgets.
- **Modèles ML** : les classes `AddressSuccessClassifier`, `ExtractionPatternLearner` et `MLParameterOptimizer` créent un dossier `models/` et sérialisent leurs artefacts (`*.pkl`). Pensez à versionner ce dossier si vous déployez sur plusieurs environnements.
- **Sorties normalisées** : les étapes génèrent des dossiers dédiés (`contacts/`, `metrics/`, `reports/`). Les fichiers existants marqués comme « frais » sont réutilisés lors des reprises.
- **Monitoring** : si vous exposez Prometheus (`--prometheus-port`), des métriques de run sont enregistrées et taguées avec le `run_id`.

## Tests et qualité

- Lancez l'ensemble de la suite :\
  `pytest`
- Exécuter un test ciblé (ex. budget middleware) :\
  `pytest tests/test_budget_middleware.py -k lines_per_s`
- La configuration `pyproject.toml` active `ruff` (linting) et `black` (formatage, 100 caractères). Respectez ces règles avant de proposer une PR.
- Les tests couvrent :
  - Middleware budgets et KPI (`tests/test_budget_middleware.py`, `tests/test_http_tracking.py`).
  - Pipeline CLI (`tests/test_builder_cli.py`, `tests/test_batch_jobs.py`).
  - Enrichissements Google Maps, parsing contacts (y compris IA), qualité et export.
  - Normalisation et génération de colonnes structurées (`tests/test_standardize.py`, `tests/test_comprehensive_extraction.py`).

Ajoutez systématiquement des tests unitaires lors de l'introduction de nouvelles étapes ou transformation d'outputs.

## Limitations connues et axes d'amélioration

- **Compatibilité des modèles persistant** (`ml/address_classifier.py:609`, `ml/extraction_models.py:554`, `utils/ml_optimizer.py:671`) : les méthodes `load_models` supposent des artefacts entraînés avec la même version de scikit-learn. Un mécanisme de versionning (hash modèle, régénération automatique) reste à implémenter.
- **KPI `lines_per_s` sensibles au contexte** (`normalize/standardize.py:826`, `utils/budget_middleware.py:118`) : si une étape ne renseigne pas `total_lines_processed`, l'indicateur tombe à zéro et déclenche de faux positifs. Une consolidation multi-étapes ou un fallback sur `rows_written` est à prévoir.
- **Gestion de répertoires de sortie** (`package/exporter.py:64`, `builder_cli.py:350`) : certaines étapes supposent l'existence de fichiers (`normalized.parquet`, dossiers `contacts/`). Un contrôle précoce de cohérence et/ou une phase de création centralisée des répertoires améliorerait la robustesse.
- **Budgets d'alerting** (`monitoring/alert_manager.py:264`) : la valeur du budget journalier est encore codée en dur (TODO). Prévoir un branchement avec `utils.cost_manager` pour automatiser les seuils.
- **Observabilité headless** : les métriques Playwright (temps de rendu, erreurs) ne sont pas encore exportées. Ajoutez des hooks Prometheus pour mieux suivre les coûts headless.

N'hésitez pas à consigner d'autres limites dans `docs/observability.md` ou dans les issues du dépôt.

## Contribuer

1. Ouvrez une issue décrivant la fonctionnalité ou le bug, en ajoutant logs et jeu de test minimal.
2. Créez une branche (`feature/`, `fix/`) et assurez-vous que `ruff` et `pytest` passent localement.
3. Documentez vos changements (README, doc dédiée, commentaires ciblés) et mettez à jour les profils/jobs si nécessaire.
4. Soumettez une Pull Request en mentionnant les tests ajoutés et les impacts potentiels (budgets, formats d'export).
5. Respectez la politique de journaux : ne committez jamais de données sensibles dans `out/` ou `models/`.

## Licence et crédits

Aucune licence explicite n'est fournie ; le code est destiné à un usage interne Infinitemoula. Contactez l'équipe Data/Automation pour clarifier les conditions de réutilisation ou pour toute publication publique.

Crédits principaux :

- Conception et développement : équipe Data Infinitemoula.
- Contributions modules scraping/API : voir l'historique Git (`git shortlog -sne`).
- Documentation : `docs/` consolidé par les squads Growth & Ops.

## Badges et visuels suggérés

Ajoutez dans ce README (ou sur la page GitHub) :

- Badge de build/CI (ex : GitHub Actions exécutant `pytest`).
- Badge de couverture (Codecov ou SonarCloud) pour suivre la qualité.
- Badge de statut Playwright (vérifie la disponibilité des navigateurs headless).
- Capture d'écran ou schéma du pipeline (ex : diagramme Mermaid dans une future section `Architecture du pipeline`).
- Badges de documentation (`docs/`) et de conformité lint (`ruff`).

## Support et roadmap

### Support

- Ouvrez une issue GitHub pour remonter un bug ou une demande produit (fournissez la commande, les logs et le contenu minimal du job).
- Pour les questions urgentes internes : contactez l'équipe Data Infinitemoula via le canal Slack `#b2b-data-pipeline` ou l'adresse `data@infinitemoula.com`.
- Consultez les guides dans [`docs/`](docs) avant de solliciter du support : beaucoup de cas (proxy, 429, gouvernance) y sont documentés.

### Roadmap (sujette à évolutions)

- Industrialisation complète des orchestrations Airflow/Dagster/Prefect (`orchestration/`).
- Déploiement d'un module de surveillance budget temps réel branché sur `utils.cost_manager`.
- Ajout de connecteurs API B2B supplémentaires (Clearbit, Pappers, LinkedIn enrichi) avec budgets dédiés.
- Amélioration du monitoring Playwright et des exports Prometheus (ajout d'étiquettes domaines, latence).
- Génération automatique d'artefacts marketing (dashboards `dashboard/`, exports enrichis segmentables).

Bonne exploration du pipeline !

