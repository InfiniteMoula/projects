# Dictionnaire de données – Contacts enrichis

| Colonne | Description | Type |
| --- | --- | --- |
| site_web | URL du site officiel identifié pour l'entité. | string |
| site_web_source | Origine de la découverte du site web (SERP, heuristique, saisie manuelle, etc.). | string |
| site_web_score | Score de confiance (0-1) attribué au site web sur la base des heuristiques. | float |
| email | Adresse email principale extraite pour l'entité. | string |
| email_source | Page ou mécanisme ayant permis de récupérer l'email. | string |
| email_score | Score de confiance (0-1) reflétant la plausibilité et la qualité de l'email. | float |
| telephone | Numéro de téléphone principal détecté. | string |
| telephone_source | Page ou mécanisme ayant permis de récupérer le numéro de téléphone. | string |
| telephone_score | Score de confiance (0-1) calculé à partir du format et du contexte du numéro. | float |
| linkedin_url | URL de la page LinkedIn de l'organisation. | string |
| linkedin_source | Source SERP ou fournisseur ayant fourni l'URL LinkedIn. | string |
| linkedin_score | Score de confiance (0-1) basé sur la similarité entre le résultat et l'entité recherchée. | float |
