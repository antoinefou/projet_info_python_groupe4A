# Prédiction du prix immobilier en France — DVF × INSEE

**Projet Python pour la Data Science — ENSAI 2024-2025 — Groupe 4A**

Yassine MELLOUL · Amira BARHOUMI · Antoine FOUCART

Chargé de TD : Julien PRAMIL

---

## Problématique

> Peut-on prédire le prix au m² des biens immobiliers en France en combinant les caractéristiques des transactions (DVF) et le contexte socio-économique local (INSEE), et quel modèle offre les meilleures performances ?

## Données

| Source | Description | Lien |
|---|---|---|
| DVF (Etalab) | Transactions immobilières en France | [data.gouv.fr](https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/) |
| INSEE — Dossier complet | Revenus, pauvreté, chômage, population par commune | [insee.fr](https://www.insee.fr/fr/statistiques/5359146) |
| Data.gouv.fr | Taux de logements sociaux par commune | [data.gouv.fr](https://www.data.gouv.fr/datasets/taux-de-logements-sociaux-dans-les-communes) |

Les données sont téléchargées automatiquement par le code — aucun fichier à stocker manuellement.

## Structure du projet

```
├── notebooks/
│   └── notebook1.ipynb          # Rapport principal (notebook)
├── src/
│   ├── data/
│   │   ├── collect.py           # Collecte des données (DVF, INSEE, logements sociaux)
│   │   ├── clean.py             # Nettoyage, feature engineering, fusion
│   │   ├── stats_desc.py        # Fonctions de statistiques descriptives
│   │   └── stat_insee.ipynb     # Exploration des données INSEE
│   └── model/
│       ├── evaluate.py          # Entraînement et évaluation des modèles
│       └── train.py             # (réservé)
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/votre-repo/projet_info_python_groupe4A.git
cd projet_info_python_groupe4A
pip install -r requirements.txt
```

## Utilisation

Lancer le notebook principal :

```bash
jupyter notebook notebooks/notebook1.ipynb
```

Ou lancer la modélisation directement :

```bash
cd src
python model/evaluate.py
```

## Méthodologie

### Collecte et nettoyage
- Téléchargement automatique des 3 sources via URL/API
- Filtrage : ventes uniquement, maisons et appartements, prix et surfaces cohérents
- Jointure sur le code commune (5 caractères INSEE)

### Variables

**Variable cible** : prix au m² (`valeur_foncière / surface_réelle_bâti`)

**Variables explicatives** :

| Variable | Source |
|---|---|
| Surface réelle bâti | DVF |
| Nombre de pièces principales | DVF |
| Type de local (Maison / Appartement) | DVF |
| Année de mutation | DVF |
| Revenu médian | INSEE |
| Taux de pauvreté | INSEE |
| Population | INSEE |
| Taux de chômage | INSEE |
| Taux de logements sociaux | data.gouv.fr |

### Modèles
1. **Régression linéaire** — modèle de référence (baseline)
2. **Random Forest** — modèle principal

### Évaluation
- Split train/test : 80/20
- Métriques : MAE, RMSE, R²

## Résultats

| Modèle | MAE (€/m²) | RMSE (€/m²) | R² |
|---|---|---|---|
| Régression linéaire | 1 659 | 2 562 | 0.29 |
| Random Forest | 1 217 | 2 090 | 0.53 |

Le Random Forest surpasse la régression linéaire, confirmant la nature non-linéaire de la relation entre les prix immobiliers et les variables explicatives.

## Limites

- Les données INSEE sont agrégées au niveau communal — la micro-localisation (quartier, rue) n'est pas captée
- DVF ne contient pas l'étage, l'état du bien, ni la présence d'extérieur
- Le taux de pauvreté présente des valeurs manquantes pour les petites communes (secret statistique INSEE)
