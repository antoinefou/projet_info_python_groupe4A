"""Évaluation et comparaison des modèles"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data.collect import load__data_url_zip_txt, load_insee_dossier_complet, url_dvf, url_insee
from data.clean import (
    filter_dvf_columns, compute_prix_m2,
    preprocess_insee, merge_dvf_insee,
    load_and_merge_logements_sociaux
)
from model.train import prepare_features, split_data, train_linear_regression, train_random_forest


def evaluate_model(y_test, y_pred, model_name):
    """Calcule et affiche les métriques d'un modèle."""
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== {model_name} ===")
    print(f"MAE  : {mae:.2f} €/m²")
    print(f"RMSE : {rmse:.2f} €/m²")
    print(f"R²   : {r2:.4f}")

    return {"Modèle": model_name, "MAE (€/m²)": mae, "RMSE (€/m²)": rmse, "R²": r2}


def plot_feature_importance(model, columns):
    """Affiche l'importance des variables du Random Forest."""
    importances = pd.DataFrame({
        "variable": columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importances["variable"], importances["importance"], color="steelblue")
    plt.title("Importance des variables - Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def plot_predictions(y_test, y_pred_reg, y_pred_rf, r2_reg, r2_rf):
    """Affiche prédictions vs réalité pour les deux modèles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_test, y_pred_reg, alpha=0.3, s=10)
    axes[0].plot([0, 15000], [0, 15000], color="red", linestyle="--")
    axes[0].set_title(f"Régression Linéaire (R² = {r2_reg:.3f})")
    axes[0].set_xlabel("Prix réel (€/m²)")
    axes[0].set_ylabel("Prix prédit (€/m²)")

    axes[1].scatter(y_test, y_pred_rf, alpha=0.3, s=10)
    axes[1].plot([0, 15000], [0, 15000], color="red", linestyle="--")
    axes[1].set_title(f"Random Forest (R² = {r2_rf:.3f})")
    axes[1].set_xlabel("Prix réel (€/m²)")
    axes[1].set_ylabel("Prix prédit (€/m²)")

    plt.tight_layout()
    plt.show()


# =============================================================
# EXÉCUTION
# =============================================================

if __name__ == "__main__":

    # 1. Charger les données
    dvf = filter_dvf_columns(load__data_url_zip_txt(url_dvf))
    dvf = compute_prix_m2(dvf)
    insee = preprocess_insee(load_insee_dossier_complet(url_insee))
    df = merge_dvf_insee(dvf, insee)
    df = load_and_merge_logements_sociaux(df)
    df["taux_chomage"] = (df["nbr_chomeur_15_64"] / df["nbr_personnes_active_15_64"]) * 100
    df["mediane_niveau_vie"] = pd.to_numeric(df["mediane_niveau_vie"], errors="coerce")
    df = df[(df["prix_m2"] > 500) & (df["prix_m2"] < 15000)]

    print(f"Dataset final : {df.shape}")

    # 2. Préparer
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

    # 3. Entraîner
    reg = train_linear_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    # 4. Évaluer
    y_pred_reg = reg.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    res_reg = evaluate_model(y_test, y_pred_reg, "Régression Linéaire")
    res_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")

    # 5. Comparaison
    resultats = pd.DataFrame([res_reg, res_rf])
    print("\n=== Comparaison ===")
    print(resultats.to_string(index=False))

    # 6. Graphiques
    plot_feature_importance(rf, X.columns)
    plot_predictions(y_test, y_pred_reg, y_pred_rf, res_reg["R²"], res_rf["R²"])

