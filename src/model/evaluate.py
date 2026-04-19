"""Modélisation et évaluation des modèles de prédiction du prix au m²"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.collect import load__data_url_zip_txt, load_insee_dossier_complet, url_dvf, url_insee
from data.clean import (
    filter_dvf_columns,
    compute_prix_m2,
    preprocess_insee,
    merge_dvf_insee,
    load_and_merge_logements_sociaux
)

# =============================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================

# Charger DVF
dvf = filter_dvf_columns(load__data_url_zip_txt(url_dvf))
dvf = compute_prix_m2(dvf)

# Charger INSEE
insee = preprocess_insee(load_insee_dossier_complet(url_insee))

# Fusionner DVF + INSEE
df = merge_dvf_insee(dvf, insee)
#test
print("DVF code_commune exemple :", dvf["code_commune"].head(5).tolist())
print("INSEE code_commune exemple :", insee["code_commune"].head(5).tolist())
print("DVF codes uniques :", dvf["code_commune"].nunique())
print("INSEE codes uniques :", insee["code_commune"].nunique())
print(df[["taux_pauvrete"]].head(10))
print(df["taux_pauvrete"].isnull().sum(), "/", len(df))

# Ajouter logements sociaux
df = load_and_merge_logements_sociaux(df)

# Calculer le taux de chômage
df["taux_chomage"] = (df["nbr_chomeur_15_64"] / df["nbr_personnes_active_15_64"]) * 100

# Convertir mediane_niveau_vie en numérique (renommé en string dans preprocess_insee)
df["mediane_niveau_vie"] = pd.to_numeric(df["mediane_niveau_vie"], errors="coerce")

# Supprimer les lignes avec des valeurs manquantes ou aberrantes
df = df.dropna(subset=["prix_m2", "mediane_niveau_vie", "taux_pauvrete", "population", "taux_chomage"])
df = df[(df["prix_m2"] > 500) & (df["prix_m2"] < 20000)]

print(f"Taille du dataset final : {df.shape}")

# =============================================================
# 2. DÉFINITION DES VARIABLES
# =============================================================

# Variable cible
y = df["prix_m2"]

# Variables explicatives numériques
features_num = [
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "annee_mutation",
    "mediane_niveau_vie",
    "taux_pauvrete",
    "population",
    "taux_chomage",
    "taux_logements_sociaux"
]

# One-hot encoding de type_local
X = pd.get_dummies(df[features_num + ["type_local"]], columns=["type_local"], drop_first=True)
X = X.apply(pd.to_numeric, errors="coerce")
print(X.dtypes)
print(X.isnull().sum())
X = X.dropna()
y = y.loc[X.index]

print(f"Variables explicatives : {X.columns.tolist()}")
print(f"X : {X.shape}, y : {y.shape}")

# =============================================================
# 3. SPLIT TRAIN / TEST
# =============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train : {X_train.shape[0]} lignes")
print(f"Test  : {X_test.shape[0]} lignes")

# =============================================================
# 4. RÉGRESSION LINÉAIRE
# =============================================================

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)

mae_reg = mean_absolute_error(y_test, y_pred_reg)
rmse_reg = np.sqrt(mean_squared_error(y_test, y_pred_reg))
r2_reg = r2_score(y_test, y_pred_reg)

print("\n=== Régression Linéaire ===")
print(f"MAE  : {mae_reg:.2f} €/m²")
print(f"RMSE : {rmse_reg:.2f} €/m²")
print(f"R²   : {r2_reg:.4f}")

# Coefficients
coefs = pd.DataFrame({
    "variable": X.columns,
    "coefficient": reg.coef_
}).sort_values("coefficient", ascending=False)
print("\nCoefficients :")
print(coefs.to_string(index=False))

# =============================================================
# 5. RANDOM FOREST
# =============================================================

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\n=== Random Forest ===")
print(f"MAE  : {mae_rf:.2f} €/m²")
print(f"RMSE : {rmse_rf:.2f} €/m²")
print(f"R²   : {r2_rf:.4f}")

# =============================================================
# 6. COMPARAISON DES MODÈLES
# =============================================================

resultats = pd.DataFrame({
    "Modèle": ["Régression Linéaire", "Random Forest"],
    "MAE (€/m²)": [mae_reg, mae_rf],
    "RMSE (€/m²)": [rmse_reg, rmse_rf],
    "R²": [r2_reg, r2_rf]
})
print("\n=== Comparaison ===")
print(resultats.to_string(index=False))

# =============================================================
# 7. IMPORTANCE DES VARIABLES (RANDOM FOREST)
# =============================================================

importances = pd.DataFrame({
    "variable": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importances["variable"], importances["importance"], color="steelblue")
plt.title("Importance des variables - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# =============================================================
# 8. VISUALISATION PRÉDICTIONS VS RÉALITÉ
# =============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Régression linéaire
axes[0].scatter(y_test, y_pred_reg, alpha=0.3, s=10)
axes[0].plot([0, 15000], [0, 15000], color="red", linestyle="--")
axes[0].set_title(f"Régression Linéaire (R² = {r2_reg:.3f})")
axes[0].set_xlabel("Prix réel (€/m²)")
axes[0].set_ylabel("Prix prédit (€/m²)")

# Random Forest
axes[1].scatter(y_test, y_pred_rf, alpha=0.3, s=10)
axes[1].plot([0, 15000], [0, 15000], color="red", linestyle="--")
axes[1].set_title(f"Random Forest (R² = {r2_rf:.3f})")
axes[1].set_xlabel("Prix réel (€/m²)")
axes[1].set_ylabel("Prix prédit (€/m²)")

plt.tight_layout()
plt.show()
