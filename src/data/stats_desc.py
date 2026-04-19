""" Statistiques descriptives univariées et bivariées"""
import pandas as pd
import matplotlib.pyplot as plt


def univariate_numeric_analysis(
    df: pd.DataFrame,
    col: str,
    bins: int = 50
):
    """
    Calcule et affiche les statistiques descriptives d'une variable numérique
    + histogramme.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.
    col : str
        Nom de la colonne numérique à analyser.
    bins : int
        Nombre de classes pour l'histogramme.

    Returns
    -------
    dict
        Dictionnaire contenant les statistiques descriptives.
    """

    x = df[[col]]

    # Statistiques
    stats = {
        "mean": x.mean(),
        "std": x.std(),
        "min": x.min(),
        "max": x.max(),
        "median": x.median(),
        "q1": x.quantile(0.25),
        "q3": x.quantile(0.75),
    }

    # Affichage stats
    print(f"Analyse univariée : {col}")
    print("-" * 40)
    for k, v in stats.items():
        print(f"{k} : {v:,.2f}")

    # Histogramme
    plt.figure(figsize=(10, 5))
    plt.hist(x, bins=bins, edgecolor="black", alpha=0.7)
    plt.title(f"Histogramme de {col}")
    plt.xlabel(col)
    plt.ylabel("Fréquence")

    # Lignes statistiques importantes
    plt.axvline(stats["mean"], color="red", linestyle="dashed", label="mean")
    plt.axvline(stats["p95"], color="orange", linestyle="dotted", label="p95")

    plt.legend()
    plt.show()
