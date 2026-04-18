"""traitement et nettoyage des données brutes"""

import pandas as pd
"""from collect import url_dvf, load__data_url_zip_txt"""


def filter_dvf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre un DataFrame DVF (Demandes de Valeurs Foncières)
    pour ne conserver que les variables pertinentes pour la
    modélisation du prix immobilier.

    Cette fonction supprime les colonnes d'identifiants,
    les variables redondantes ou trop fines, et conserve les
    variables explicatives utiles liées à :

    - la localisation
    - les caractéristiques du bien
    - le type de transaction

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame brut issu des données DVF contenant toutes
        les colonnes initiales (brutes, non nettoyées).

    Returns
    -------
    pd.DataFrame
        DataFrame filtré ne contenant que les variables utiles
        à la modélisation du prix immobilier.
    """

    # Colonnes utiles (version standard projet immobilier)
    useful_cols = [
        # cible
        "Valeur fonciere",

        # localisation
        "Code postal",
        "Commune",
        "Code departement",
        "Code commune",

        # caractéristiques bien
        "Type local",
        "Surface reelle bati",

        # information type vente
        "Nature mutation"
    ]

    # duplicated?

    # garder uniquement les colonnes existantes (robustesse)
    cols_to_keep = [col for col in useful_cols if col in df.columns]

    df_filtered = df[cols_to_keep].copy()

    df_filtered["Valeur fonciere"] = (
        df_filtered["Valeur fonciere"]
        .str.replace(" ", "")
        .str.replace(",", ".")
        .astype(float))

    df_filtered = df_filtered.astype({
        "Commune": "string",
        "Code departement": "string",
        "Code commune": "string",
        "Type local": "string",
        "Nature mutation": "string"
    })

    df_filtered = df_filtered[
        (df_filtered["Nature mutation"] == "Vente") &
        (df_filtered["Valeur fonciere"] > 0) &
        (df_filtered["Surface reelle bati"] > 0) &
        (df_filtered["Type local"].isin(["Maison", "Appartement"]))
    ]

    df_filtered = df_filtered.dropna()

    df_filtered["Code postal"] = (
        df_filtered["Code postal"]
        .astype(int)
        .astype(str)
        )

    return df_filtered


"""dvf = filter_dvf_columns(load__data_url_zip_txt(url_dvf))
print(dvf.dtypes)
print(dvf)"""


def compute_prix_m2(
    df: pd.DataFrame,
    price_col: str = "Valeur fonciere",
    surface_col: str = "Surface reelle bati",
    new_col: str = "prix_m2"
) -> pd.DataFrame:
    """
    Calcule le prix au m² à partir d'un DataFrame DVF.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données DVF.
    price_col : str
        Colonne du prix total (par défaut "Valeur fonciere").
    surface_col : str
        Colonne de surface bâtie (par défaut "Surface reelle bati").
    new_col : str
        Nom de la colonne de sortie (prix au m²).

    Returns
    -------
    pd.DataFrame
        DataFrame avec une nouvelle colonne prix au m².
    """

    df = df.copy()
    # Calcul du prix au m²
    df[new_col] = df[price_col] / df[surface_col]

    return df
