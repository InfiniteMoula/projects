# Migration vers la v2 du pipeline

Ce guide détaille les actions à mener pour migrer une instance existante du pipeline vers la version 2.0. Il couvre la mise à jour de l'environnement, l'adaptation des workflows et les contrôles de validation recommandés.

## 1. Préparer l'environnement

1. **Geler la production actuelle** : terminez les runs en cours et archivez les sorties critiques (`normalized.parquet`, `contacts/`, exports finaux) avant d'initier la migration.
2. **Mettre à jour les dépendances** :
   - Installez Python 3.11 si ce n'est pas déjà le cas.
   - Exécutez `pip install -r requirements.txt --upgrade` dans un nouvel environnement virtuel.
   - Déployez Playwright si vous utilisez les étapes headless : `python -m playwright install`.
3. **Synchroniser la configuration** : fusionnez vos surcharges locales avec les nouveaux exemples présents dans `jobs/` et `config/`. Les profils v2 introduisent de nouveaux paramètres (`standardize_batch_rows`, `quality.email_generic_list`).

## 2. Adapter les workflows

1. **Scripts utilitaires** :
   - Utilisez `scripts/run_tests.sh` (Linux/macOS) ou `scripts/run_tests.bat` (Windows) pour vérifier la conformité qualité (`pytest -q`, `ruff`, `mypy`).
   - Remplacez tout appel aux anciens scripts customisés par les commandes du CLI (`builder_cli.py`).
2. **Normalisation SIRENE** : le step `normalize.standardize` extrait désormais l'intégralité des colonnes disponibles et calcule automatiquement les colonnes structurées (téléphone normalisé, département, etc.). Assurez-vous que vos pipelines down-stream consomment les nouveaux noms de colonnes (`departement`, `Téléphone standard`, `Nom entreprise`).
3. **Budgets & KPI** : vérifiez vos fichiers de job et ajoutez les nouveaux seuils si besoin (`kpi_targets.min_lines_per_s`, budgets HTTP détaillés). Les profils `standard` et `deep` appliquent désormais des contrôles plus stricts.
4. **Orchestration** : si vous pilotez le pipeline via Airflow/Dagster/Prefect, mettez à jour les DAG/flows pour refléter les nouveaux arguments CLI (`--standardize-batch-rows`, `--quality-config`).

## 3. Vérifications post-migration

1. **Exécuter un run de validation** sur un échantillon réduit (`--sample 50`) pour contrôler :
   - La génération des rapports `out/<job>/reports/standardize_summary.json`.
   - La présence des jeux `contacts/contacts.parquet` et `normalized.parquet`.
   - Le respect des KPI configurés (`rows_per_s`, `quality_score`).
2. **Contrôles qualité** : lancez `scripts/run_tests.sh` et confirmez que `pytest`, `ruff` et `mypy --ignore-missing-imports` passent sans erreur.
3. **Monitoring** : si vous exposez des métriques Prometheus, vérifiez que les labels supplémentaires (`profile`, `naf_code`) apparaissent dans vos dashboards.
4. **Documentation interne** : mettez à jour vos playbooks opérationnels et informez les équipes Data/Growth des changements (nouvelles colonnes, scripts de test, seuils KPI).

Une fois ces étapes validées, vous pouvez décommissionner l'ancienne version et reprendre vos runs de production en v2.
