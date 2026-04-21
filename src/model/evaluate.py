"""Fonctions d'évaluation et de visualisation des modèles"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def evaluate_model(y_test, y_pred, model_name):
    """
    Calcule et affiche les métriques d'un modèle.

    Parameters
    ----------
    y_test : array-like
        Valeurs réelles
    y_pred : array-like
        Valeurs prédites
    model_name : str
        Nom du modèle

    Returns
    -------
    dict
        Dictionnaire des métriques
    """
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


def plot_predictions(y_test, predictions, model_names):
    """
    Affiche prédictions vs réalité pour plusieurs modèles.

    Parameters
    ----------
    y_test : array-like
        Valeurs réelles
    predictions : list of array-like
        Liste des prédictions de chaque modèle
    model_names : list of str
        Noms des modèles
    """
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))

    if n == 1:
        axes = [axes]

    for ax, y_pred, name in zip(axes, predictions, model_names):
        r2 = r2_score(y_test, y_pred)
        ax.scatter(y_test, y_pred, alpha=0.3, s=10)
        ax.plot([0, 15000], [0, 15000], color="red", linestyle="--")
        ax.set_title(f"{name} (R² = {r2:.3f})")
        ax.set_xlabel("Prix réel (€/m²)")
        ax.set_ylabel("Prix prédit (€/m²)")

    plt.tight_layout()
    plt.show()


def plot_residuals(y_test, y_pred, model_name="Modèle"):
    """Affiche la distribution des résidus."""
    residus = y_test - y_pred

    plt.figure(figsize=(10, 5))
    plt.hist(residus, bins=50, edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--")
    plt.title(f"Distribution des résidus - {model_name}")
    plt.xlabel("Erreur (€/m²)")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.show()
