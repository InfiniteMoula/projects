# Tâche : Script de filtrage des lignes vendables

## Contexte
Le pipeline produit un gros fichier d'entreprises enrichies (CSV ou Parquet).
Toutes les lignes ne sont pas intéressantes commercialement.

Objectif :
Créer un script Python qui prend en entrée un CSV ou Parquet d'entreprises enrichies
et écrit en sortie un fichier ne contenant que les **lignes vendables** (définition dans docs/products.md).

## Fichiers concernés
- Nouveau fichier : `scripts/filter_vendable_rows.py`

## Entrées attendues
- Fichier input (obligatoire) : chemin vers un fichier `.csv` ou `.parquet`
- Colonnes minimum attendues :
  - `siret`, `nom`, `code_naf`
  - `ville` (ou `city` ou équivalent)
  - `domain`
  - `email` (optionnel)
  - `phone` ou `telephone` (optionnel)

## Règles de filtrage (version 1)
Une ligne est gardée si :
- `siret`, `nom`, `code_naf` ne sont pas nuls/vides
- `domain` est non nul et non vide
- au moins une des colonnes suivantes est non nulle :
  - `email`
  - `phone` ou `telephone`

Les noms de colonnes doivent être configurables facilement dans le code
(ex : un petit dict de mapping).

## Output attendu (pour l'IA)
- Code complet de `scripts/filter_vendable_rows.py`
- Utilisation :
  - `python scripts/filter_vendable_rows.py --input input.csv --output output.csv`
  - ou avec `.parquet` en entrée
- Logs :
  - nombre de lignes en entrée
  - nombre de lignes vendables en sortie
  - pourcentage de lignes gardées

## Contraintes
- Python 3.11
- Utiliser `pandas`
- Utiliser `logging` (pas de `print`)
- Code simple à lire et modifier
