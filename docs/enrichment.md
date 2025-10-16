# Zero-capital enrichment workflow

Ce document resume les modules d'enrichissement zero-capital, explique la configuration `config/enrichment.yaml`, et decrit comment lancer le profil `standard_nocapital`.

## Modules d'enrichissement

- `enrich.domains`  
  Identifie le(s) domaine(s) web officiel(s) a partir du dataset normalise. Le module combine des requetes SERP (Bing, DuckDuckGo par defaut) et une exploration heuristique des TLD et prefixes (`www.`) pour confirmer un site. Le traitement est parallele par blocs pour accelerer les jeux de donnees volumineux tout en respectant les limites de requetes (`RequestLimiter` partage).

- `enrich.contacts`  
  Explore les sites identifies afin d'extraire emails et numeros de telephone. La recherche repose sur des chemins favoris (contact, mentions legales, privacy...), peut suivre sitemap et robots.txt, et applique des heuristiques pour scorer les candidats (emails nominatif vs generique, type de numero, concordance ville). Les resultats sont exportes dans `contacts_enriched.{parquet,csv}`.

- `enrich.linkedin`
  Cherche la page LinkedIn d'une entite sans gratter LinkedIn directement. Le module formule des requetes `site:linkedin.com/company` via les fournisseurs SERP configures puis score les resultats sur la similarite avec la denomination normalisee. Les correspondances retenues sont exportees dans `linkedin_enriched.{parquet,csv}` avec la source SERP et un score de confiance.

- `enrich.linkedin_clearbit_lite`
  Produit un enrichissement "LinkedIn & Clearbit" purement heuristique pour les profils standard et deep. A partir des domaines detectes, des mots-clefs et des codes NAF, le module simule `industry`, `employee_range` et `linkedin_url` puis enregistre le resultat dans `linkedin_clearbit_lite.{parquet,csv}`.

Chaque module ecrit ses sorties dans le repertoire `outdir` du run et respecte les drapeaux `use_domains`, `use_contacts`, `use_linkedin` de la configuration.

## Configuration `config/enrichment.yaml`

### Drapeaux generaux

- `use_domains`, `use_contacts`, `use_linkedin` : activent ou non chaque module lors des executions CLI. Les options `--use-*` de `builder_cli` peuvent les surcharger ponctuellement.

### Bloc `domains`

- `providers` / `providers_config` : liste et parametrage des fournisseurs SERP utilises pour decouvrir les domaines (ex. limiter `max_results`).
- `http_client` : configuration reseau partagee (voir ci-dessous). Les champs essentiels :
  - `user_agents_file` ou `user_agents` : pool de User-Agent a alterner (obligatoire pour eviter les blocages).
  - `max_concurrent_requests` : plafond global de requetes simultanees (impacter la charge minute).
  - `per_host_limit` : limite par hostname pour rester courtois.
  - `timeout` : delai maximal avant abandon d'une requete.
  - `retry_attempts` / `delay_base` : repetition avec backoff exponentiel.
  - `cache_dir` / `cache_ttl_days` : cache HTTP persistant et sa duree de vie.
- `serp_score_threshold` : score minimal pour accepter un resultat SERP comme domaine officiel.
- `heuristic_score_threshold` : score minimal pour les tentatives heuristiques (ex. slug+tld).
- `tlds`, `prefixes` : suffixes/prefixes explores lorsqu'on tente des combinaisons (les points initiaux sont acceptes dans le YAML pour la lisibilite).
- `extra_generic_domains` : permet d'ajouter des domaines a exclure des SERP en plus de ceux definis dans `constants/generic_domains.txt` (annuaires, reseaux sociaux, etc.).

### Bloc `contacts`

- `http_client` : meme structure que ci-dessus; reutilise typiquement le meme pool d'agents et la meme politique de cache.
- `pages_to_scan` : chemins prioritaires a collecter sur chaque site.
- `max_pages_per_site` : coupe la collecte pour eviter de s'etendre sur de grands sites.
- `sitemap_limit` : nombre maximum d'URLs ajoutees depuis le sitemap.
- `use_sitemap`, `use_robots` : activent la decouverte par sitemap et le respect de robots.txt.
- `email_generic_domains` / `email_generic_prefixes` : listes pour deprioriser les emails trop generiques. Des valeurs par defaut sont chargees depuis `constants/generic_email_domains.txt` et `constants/generic_email_prefixes.txt` puis enrichies avec celles declarees dans la configuration.

### Bloc `linkedin`

- `providers` / `providers_config` : identiques au bloc domains, mais appliques aux recherches LinkedIn.
- `http_client` : meme schema que precedent; souvent partage un cache dedie dans `.cache/http`.

## Lancer le profil `standard_nocapital`

1. **Prerequis**  
   - Installer les dependances : `pip install -r requirements.txt` dans l'environnement de travail.  
   - Preparer un fichier de pool User-Agent a `data/user_agents.txt` (une ligne par UA) ou renseigner la liste inline dans `config/enrichment.yaml`.  
   - Verifier la presence de `config/enrichment.yaml` (celui du depot contient des valeurs par defaut adaptees au zero-capital).
2. **Preparation du job**  
   - Creer ou reutiliser un fichier job YAML (ex. `jobs/naf_6202A.yaml`) definissant la source de donnees et le profil souhaite.  
   - Optionnel : forcer `use_domains`, `use_contacts`, `use_linkedin` via la CLI si besoin de dev tests.
3. **Commande**  
   ```bash
   python builder_cli.py run-profile \
     --job jobs/naf_6202A.yaml \
     --out out/naf_6202A_zero_capital \
     --profile standard_nocapital
   ```
   Parametres utiles : `--resume` pour reprendre un run, `--dry-run` pour afficher le plan d'execution, `--verbose` ou `--debug` pour plus de logs.

Les sorties de ce profil incluent `domains_enriched.*`, `contacts_enriched.*`, `linkedin_enriched.*` et le dataset final `package.export`.

## Recommandations d'ajustement

- **Debit de requetes** : reduire `http_client.max_concurrent_requests` ou `per_host_limit` si vous observez des HTTP 429, ou augmenter doucement pour des lots plus grands (garder un ratio RPS raisonnable).  
- **TTL de cache** : utiliser `cache_ttl_days` pour maitriser la fraicheur des resultats SERP ou des pages contacts; augmenter pour des runs repetitifs sur un meme secteur, reduire pour des secteurs tres dynamiques.  
- **Chunking & workers** (`enrich.domains` uniquement) : ajuster `chunk_size` et `max_workers` via le contexte CLI (arguments `--workers`) ou la config si les datasets sont tres volumineux.  
- **Pages contacts** : etendre `pages_to_scan` pour couvrir des patterns specifiques a un secteur (ex. `/coordonnees`, `/nous-trouver`).  
- **Providers SERP** : vous pouvez desactiver Bing ou DuckDuckGo, ou ajouter un fournisseur custom implemente dans `serp.providers`.

## Limites connues et pistes d'amelioration

- **Variabilite SERP** : les providers peuvent retourner des resultats bruites ou se bloquer apres plusieurs requetes.  
  *Pistes* : etendre le pool de User-Agent, ajouter d'autres moteurs, mettre en place un monitoring des taux de reussite par provider.
- **Sites minimalistes** : de nombreuses TPE n'ont pas de page contact claire ni de mentions legales; le module contacts ne trouve alors que des emails generiques ou rien.  
  *Pistes* : integrer des sources tierces (annuaires sectoriels) ou utiliser un fallback headless cible.
- **Score LinkedIn** : les resultats de recherche peuvent referencer des succursales ou des marques proches.  
  *Pistes* : enrichir le scoring (ville, code postal) et ajouter une validation humaine sur les nouveaux noms.
- **Couverture internationale** : les TLD et prefixes actuels sont calibres pour la France/Europe.  
  *Pistes* : ajuster `tlds` et `prefixes`, et introduire des providers regionaux lorsque necessaire.

Pour toute question ou contribution, ouvrez une issue ou consultez `ENRICHMENT_IMPLEMENTATION.md` pour plus de details techniques.

