"""Statistiques descriptives univariées et bivariées."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def _check_column(df: pd.DataFrame, col: str) -> None:
    """Vérifie qu'une colonne existe dans le DataFrame."""
    if col not in df.columns:
        raise ValueError(f"La colonne '{col}' est absente du DataFrame.")


def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Convertit une colonne en série numérique exploitable."""
    _check_column(df, col)
    x = pd.to_numeric(df[col], errors="coerce").dropna()

    if x.empty:
        raise ValueError(
            f"La colonne '{col}' ne contient aucune valeur numérique exploitable."
        )
    return x


def numeric_summary(df: pd.DataFrame, col: str) -> dict:
    """
    Retourne les statistiques descriptives principales d'une variable numérique.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.
    col : str
        Nom de la colonne numérique.

    Returns
    -------
    dict
        Dictionnaire de statistiques.
    """
    x = _to_numeric_series(df, col)

    stats = {
        "count": int(x.count()),
        "missing": int(df[col].isna().sum()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "p05": float(x.quantile(0.05)),
        "q1": float(x.quantile(0.25)),
        "median": float(x.median()),
        "q3": float(x.quantile(0.75)),
        "p95": float(x.quantile(0.95)),
        "max": float(x.max()),
        "iqr": float(x.quantile(0.75) - x.quantile(0.25)),
        "skew": float(x.skew()),
    }

    return stats


def univariate_numeric_analysis(
    df: pd.DataFrame,
    col: str,
    bins: int = 50,
    figsize: tuple[int, int] = (10, 5),
    show_plot: bool = True,
) -> dict:
    """
    Analyse univariée d'une variable numérique :
    - statistiques descriptives
    - histogramme
    - lignes de repère (moyenne, médiane, p95)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.
    col : str
        Nom de la colonne numérique.
    bins : int, default=50
        Nombre de classes pour l'histogramme.
    figsize : tuple, default=(10, 5)
        Taille de la figure.
    show_plot : bool, default=True
        Affiche ou non le graphique.

    Returns
    -------
    dict
        Statistiques descriptives.
    """
    stats = numeric_summary(df, col)

    print(f"\nAnalyse univariée numérique : {col}")
    print("-" * 50)
    for key, value in stats.items():
        if key in {"count", "missing"}:
            print(f"{key:<10} : {int(value)}")
        else:
            print(f"{key:<10} : {value:,.2f}")

    if show_plot:
        x = _to_numeric_series(df, col)

        plt.figure(figsize=figsize)
        plt.hist(x, bins=bins, edgecolor="black", alpha=0.7)
        plt.axvline(
            stats["mean"], linestyle="dashed", linewidth=2, label="Moyenne"
        )
        plt.axvline(
            stats["median"], linestyle="solid", linewidth=2, label="Médiane"
        )
        plt.axvline(
            stats["p95"], linestyle="dotted", linewidth=2, label="P95"
        )
        plt.title(f"Histogramme de {col}")
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return stats


def univariate_categorical_analysis(
    df: pd.DataFrame,
    col: str,
    top_n: int | None = 10,
    figsize: tuple[int, int] = (10, 5),
    show_plot: bool = True,
) -> pd.DataFrame:
    """
    Analyse univariée d'une variable catégorielle :
    - effectifs
    - proportions
    - diagramme en barres

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.
    col : str
        Nom de la colonne catégorielle.
    top_n : int | None, default=10
        Nombre de modalités à afficher dans le graphique.
        Si None, affiche toutes les modalités.
    figsize : tuple, default=(10, 5)
        Taille de la figure.
    show_plot : bool, default=True
        Affiche ou non le graphique.

    Returns
    -------
    pd.DataFrame
        Tableau des effectifs et proportions.
    """
    _check_column(df, col)

    counts = (
        df[col]
        .astype("string")
        .fillna("Valeur manquante")
        .value_counts(dropna=False)
    )
    proportions = (counts / counts.sum() * 100).round(2)

    result = pd.DataFrame(
        {
            "modalite": counts.index.astype(str),
            "effectif": counts.values,
            "proportion_%": proportions.values,
        }
    )

    print(f"\nAnalyse univariée catégorielle : {col}")
    print("-" * 50)
    print(result.head(top_n if top_n is not None else len(result)))

    if show_plot:
        plot_df = result.head(top_n) if top_n is not None else result.copy()

        plt.figure(figsize=figsize)
        plt.bar(plot_df["modalite"], plot_df["effectif"])
        plt.title(f"Répartition de {col}")
        plt.xlabel(col)
        plt.ylabel("Effectif")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    return result


def bivariate_numeric_numeric_analysis(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    sample_size: int | None = 5000,
    figsize: tuple[int, int] = (7, 5),
    show_plot: bool = True,
) -> dict:
    """
    Analyse bivariée entre deux variables numériques :
    - corrélation de Pearson
    - nuage de points

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.
    x_col : str
        Variable explicative.
    y_col : str
        Variable cible ou variable à comparer.
    sample_size : int | None, default=5000
        Taille de l'échantillon pour le scatter plot.
        Si None, utilise toutes les lignes.
    figsize : tuple, default=(7, 5)
        Taille de la figure.
    show_plot : bool, default=True
        Affiche ou non le graphique.

    Returns
    -------
    dict
        Résultats de l'analyse.
    """
    _check_column(df, x_col)
    _check_column(df, y_col)

    temp = df[[x_col, y_col]].copy()
    temp[x_col] = pd.to_numeric(temp[x_col], errors="coerce")
    temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce")
    temp = temp.dropna()

    if temp.empty:
        raise ValueError("Aucune donnée exploitable après conversion numérique.")

    if sample_size is not None and len(temp) > sample_size:
        temp_plot = temp.sample(sample_size, random_state=42)
    else:
        temp_plot = temp

    corr = float(temp[x_col].corr(temp[y_col]))

    result = {
        "n_obs": int(len(temp)),
        "correlation_pearson": corr,
        "x_mean": float(temp[x_col].mean()),
        "y_mean": float(temp[y_col].mean()),
    }

    print(f"\nAnalyse bivariée numérique-numérique : {x_col} ~ {y_col}")
    print("-" * 50)
    print(f"Nombre d'observations : {result['n_obs']}")
    print(f"Corrélation de Pearson : {result['correlation_pearson']:.4f}")

    if show_plot:
        plt.figure(figsize=figsize)
        plt.scatter(temp_plot[x_col], temp_plot[y_col], alpha=0.35)
        plt.title(f"{y_col} en fonction de {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        plt.show()

    return result


def bivariate_numeric_categorical_analysis(
    df: pd.DataFrame,
    num_col: str,
    cat_col: str,
    top_n: int = 10,
    figsize: tuple[int, int] = (11, 5),
    show_plot: bool = True,
) -> pd.DataFrame:
    """
    Analyse bivariée entre une variable numérique et une variable catégorielle :
    - statistiques par groupe
    - boxplot sur les top_n catégories les plus fréquentes

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.
    num_col : str
        Colonne numérique.
    cat_col : str
        Colonne catégorielle.
    top_n : int, default=10
        Nombre de catégories les plus fréquentes à représenter.
    figsize : tuple, default=(11, 5)
        Taille de la figure.
    show_plot : bool, default=True
        Affiche ou non le graphique.

    Returns
    -------
    pd.DataFrame
        Tableau des statistiques par groupe.
    """
    _check_column(df, num_col)
    _check_column(df, cat_col)

    temp = df[[num_col, cat_col]].copy()
    temp[num_col] = pd.to_numeric(temp[num_col], errors="coerce")
    temp[cat_col] = temp[cat_col].astype("string")
    temp = temp.dropna()

    if temp.empty:
        raise ValueError("Aucune donnée exploitable pour l'analyse bivariée.")

    grouped = temp.groupby(cat_col)[num_col]

    stats = (
        grouped.agg(
            effectif="count",
            moyenne="mean",
            mediane="median",
            ecart_type="std",
            minimum="min",
            maximum="max",
        )
        .sort_values("effectif", ascending=False)
    )

    stats["q1"] = grouped.quantile(0.25)
    stats["q3"] = grouped.quantile(0.75)
    stats["iqr"] = stats["q3"] - stats["q1"]

    top_stats = stats.head(top_n).copy()

    print(f"\nAnalyse bivariée numérique-catégorielle : {num_col} ~ {cat_col}")
    print("-" * 70)
    print(top_stats.round(2))

    if show_plot:
        categories = top_stats.index.tolist()
        plot_data = [
            temp.loc[temp[cat_col] == cat, num_col].dropna().values
            for cat in categories
        ]

        plt.figure(figsize=figsize)
        plt.boxplot(plot_data, labels=categories, showfliers=False)
        plt.title(f"Distribution de {num_col} par {cat_col} (top {top_n})")
        plt.xlabel(cat_col)
        plt.ylabel(num_col)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    return stats.reset_index()


def correlation_matrix(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "pearson",
    figsize: tuple[int, int] = (8, 6),
    show_plot: bool = True,
) -> pd.DataFrame:
    """
    Calcule et affiche une matrice de corrélation avec matplotlib.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.
    cols : list[str]
        Colonnes numériques à inclure.
    method : str, default="pearson"
        Méthode de corrélation.
    figsize : tuple, default=(8, 6)
        Taille de la figure.
    show_plot : bool, default=True
        Affiche ou non le graphique.

    Returns
    -------
    pd.DataFrame
        Matrice de corrélation.
    """
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes absentes : {missing_cols}")

    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method)

    print("\nMatrice de corrélation")
    print("-" * 30)
    print(corr.round(3))

    if show_plot:
        plt.figure(figsize=figsize)
        plt.imshow(corr, interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.index)), corr.index)

        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

        plt.title("Matrice de corrélation")
        plt.tight_layout()
        plt.show()

    return corr


def analyse_descriptive_complete(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> dict:
    """
    Lance une analyse descriptive standard sur un DataFrame immobilier.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame d'analyse.
    numeric_cols : list[str] | None
        Colonnes numériques à analyser.
    categorical_cols : list[str] | None
        Colonnes catégorielles à analyser.

    Returns
    -------
    dict
        Résultats structurés de l'analyse.
    """
    if numeric_cols is None:
        numeric_cols = [
            c for c in ["Valeur fonciere", "Surface reelle bati", "prix_m2"]
            if c in df.columns
        ]

    if categorical_cols is None:
        categorical_cols = [
            c for c in ["Type local", "Code departement", "Commune"]
            if c in df.columns
        ]

    results = {"numeric": {}, "categorical": {}}

    print("=" * 80)
    print("ANALYSE DESCRIPTIVE COMPLETE")
    print("=" * 80)
    print(f"Nombre de lignes : {df.shape[0]}")
    print(f"Nombre de colonnes : {df.shape[1]}")

    for col in numeric_cols:
        results["numeric"][col] = univariate_numeric_analysis(df, col, show_plot=True)

    for col in categorical_cols:
        results["categorical"][col] = univariate_categorical_analysis(df, col, show_plot=True)

    if {"Surface reelle bati", "Valeur fonciere"}.issubset(df.columns):
        results["surface_vs_valeur"] = bivariate_numeric_numeric_analysis(
            df, "Surface reelle bati", "Valeur fonciere", show_plot=True
        )

    if {"Type local", "prix_m2"}.issubset(df.columns):
        results["prix_m2_vs_type_local"] = bivariate_numeric_categorical_analysis(
            df, "prix_m2", "Type local", top_n=10, show_plot=True
        )

    return results