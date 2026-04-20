""" Statistiques descriptives univariées et bivariées"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.stats import pearsonr
import geopandas as gpd
from cartiflette import carti_download


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

    x = df[col]

    # Statistiques
    stats = {
        "mean": x.mean(),
        "std": x.std(),
        "min": x.min(),
        "max": x.max(),
        "median": x.median(),
        "q1": x.quantile(0.25),
        "q3": x.quantile(0.75),
        "q05": x.quantile(0.05),
        "q95": x.quantile(0.95)
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
    plt.axvline(stats["q1"], color="orange", linestyle="dotted", label="q1")
    plt.axvline(stats["q3"], color="orange", linestyle="dotted", label="q3")

    plt.legend()
    plt.show()

    return stats


def analyse_bivariée_quant_quant(df, cols, method="pearson"):
    """
    Analyse bivariée sur variables quantitatives choisies :
    - matrice de corrélation
    - matrice de p-values
    - nuages de points pour toutes les paires

    Parameters
    ----------
    df : pd.DataFrame
    cols : list
        colonnes à analyser (doivent être numériques)
    method : str
        uniquement 'pearson' ici

    Returns
    -------
    corr_df : pd.DataFrame
    pval_df : pd.DataFrame
    """

    # Sous-dataset
    data = df[cols].dropna()

    # vérification
    if data.shape[1] < 2:
        raise ValueError("Il faut au moins 2 variables.")

    #  Matrices vides
    corr = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pvals = pd.DataFrame(index=cols, columns=cols, dtype=float)

    # Calcul corrélation + p-value
    for i in cols:
        for j in cols:
            if i == j:
                corr.loc[i, j] = 1.0
                pvals.loc[i, j] = 0.0
            else:
                r, p = pearsonr(data[i], data[j])
                corr.loc[i, j] = r
                pvals.loc[i, j] = p

    print(" Corrélations :\n", corr, "\n")
    print(" P-values :\n", pvals, "\n")

    # Nuages de points
    pairs = list(itertools.combinations(cols, 2))
    n = len(pairs)

    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).reshape(-1)

    for k, (x, y) in enumerate(pairs):
        axes[k].scatter(data[x], data[y], alpha=0.5)

        r = corr.loc[x, y]
        p = pvals.loc[x, y]

        axes[k].set_title(f"{x} vs {y}\nr={r:.2f}, p={p:.3e}")
        axes[k].set_xlabel(x)
        axes[k].set_ylabel(y)

    # suppression axes vides
    for k in range(len(pairs), len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    plt.show()

    return corr, pvals

# d'abord test ANOVA!!!


def carte_dep_communes_cartiflette(df, col, code_dep, code_commune_col="code_commune", year=2022, simplification=50):
    """
    Carte choroplèthe des communes d'un département
    directement via cartiflette + données utilisateur.

    Parameters
    ----------
    df : pd.DataFrame
        Données contenant une variable quantitative + code commune INSEE
    col : str
        Variable à cartographier (ex: prix_m2)
    code_dep : str or int
        Code département (ex: 69, 75)
    code_commune_col : str
        Colonne code commune INSEE (5 chiffres)
    year : int
        Millésime cartiflette
    simplification : int
        Niveau de simplification géométrique

    Returns
    -------
    GeoDataFrame
    """

    # Téléchargement des communes via cartiflette
    communes = carti_download(
        values=["France"],
        crs=4326,
        borders="COMMUNE",
        vectorfile_format="geojson",
        simplification=simplification,
        filter_by="FRANCE_ENTIERE_DROM_RAPPROCHES",
        source="EXPRESS-COG-CARTO-TERRITOIRE",
        year=year
    )

    # Préparation codes
    df = df.copy()
    df[code_commune_col] = df[code_commune_col].astype(str)
    communes["INSEE_COM"] = communes["INSEE_COM"].astype(str)

    # Création code département depuis code commune
    communes["DEP"] = communes["INSEE_COM"].str[:2]

    # Filtrage département
    communes_dep = communes[communes["DEP"] == str(code_dep)]

    #  Agrégation de la variable
    df_agg = df.groupby(code_commune_col)[col].mean().reset_index()

    #  Jointure data ↔ géographie
    gdf = communes_dep.merge(
        df_agg,
        left_on="INSEE_COM",
        right_on=code_commune_col,
        how="left"
    )

    # Plot choroplèthe
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    gdf.plot(
        column=col,
        ax=ax,
        cmap="viridis",
        legend=True,
        missing_kwds={"color": "lightgrey"},
        edgecolor="white",
        linewidth=0.2
    )

    ax.set_title(f"Carte département {code_dep} - {col}", fontsize=14)
    ax.axis("off")

    plt.show()

    return gdf
