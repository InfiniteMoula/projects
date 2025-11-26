# Produits datasets

## 1. Objectif des datasets

Les datasets sont destinés à des clients B2B qui veulent :
- contacter des entreprises ciblées,
- faire de la prospection (emailing, phoning),
- analyser un segment de marché (par NAF, zone géographique, taille, etc.).

Un dataset doit donc être :
- propre,
- cohérent,
- exploitable immédiatement (sans nettoyage lourd).

---

## 2. Définition d'une "ligne vendable"

Une entreprise est considérée **vendable** si :

1. **Identification minimale fiable**
   - `siret` non nul
   - `nom` non nul
   - `code_naf` non nul

2. **Contact direct possible**
   - `domain` non nul et non vide
   - au moins une des conditions :
     - `email` non nul
     - ou `phone` (téléphone) non nul

3. **Données géographiques minimales**
   - `ville` non nulle (ou équivalent : `commune`, `city`)

Cette définition est **version 1** : elle doit être facile à faire évoluer.

---

## 3. Packs de datasets

### 3.1 Pack Quick
- Colonnes : siret, nom, code_naf, ville, domain
- Conditions "vendable" :
  - domain non nul
- Prix : bas, volume important

### 3.2 Pack Standard
- Colonnes : Quick + email + téléphone si disponibles
- Conditions "vendable" :
  - domain non nul
  - email non nul
- Prix : moyen

### 3.3 Pack Premium
- Colonnes : Standard + Google Maps (url, rating, reviews) + éventuels liens LinkedIn / réseaux
- Conditions "vendable" :
  - critères Standard
  - + au moins une info enrichie (GMaps ou autre)
- Prix : élevé

---

## 4. Règles générales

- Tous les scripts de filtrage/exports doivent se baser sur cette définition,
  et lire ce fichier comme source de vérité (docs, pas forcément en code).
- Toute modification de définition doit être notée dans `docs/ai_changelog.md`.
