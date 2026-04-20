"""traitement et nettoyage des données brutes"""

import pandas as pd
import numpy as np



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
        "Nombre pieces principales", 

        # information type vente
        "Nature mutation"
    ]

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

    
    # rename
    df_filtered = df_filtered.rename(columns={
        "Valeur fonciere": "valeur_fonciere",
        "Code postal": "code_postal",
        "Commune": "commune",
        "Code departement": "code_departement",
        "Code commune": "code_commune",
        "Type local": "type_local",
        "Surface reelle bati": "surface_reelle_bati",
        "Nature mutation": "nature_mutation",
        "Nombre pieces principales": "nombre_pieces_principales",
    })

    df_filtered["code_departement"] = df_filtered["code_departement"].astype(str).str.zfill(2)
    df_filtered["code_commune"] = df_filtered["code_commune"].astype(str).str.zfill(3)

    df_filtered["code_commune"] = df_filtered["code_departement"] + df_filtered["code_commune"]

    return df_filtered




def compute_prix_m2(
    df: pd.DataFrame,
    price_col: str = "valeur_fonciere",
    surface_col: str = "surface_reelle_bati",
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


def preprocess_insee(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et transforme un DataFrame INSEE :

    - Renomme certaines colonnes
    - Convertit les types de données

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame d'entrée

    Retour
    ------
    pd.DataFrame
        DataFrame transformé
    """

    # Copie pour éviter de modifier l'original
    df = df.copy()
    
    # 1. Renommage des colonnes
    df = df.rename(columns={
        "CODGEO": "code_commune",
        "MED21": "mediane_niveau_vie",
        "TP6021": "taux_pauvrete",
        "P22_POP": "population",
        "P22_CHOM1564": "nbr_chomeur_15_64",
        "P22_ACT1564": "nbr_personnes_active_15_64",
        "SUPERF": "superficie"
    })
    # Convertir les virgules en points pour les colonnes numériques
    for col in ["mediane_niveau_vie"]:
        df[col] = df[col].astype(str).str.replace(",", ".").apply(pd.to_numeric, errors="coerce")

    # 2. Conversion des types
    df["code_commune"] = df["code_commune"].astype("string")

    df["taux_pauvrete"] = (
        pd.to_numeric(
            df["taux_pauvrete"]
            .replace(["s", "nd"], np.nan)
            .str.replace(",", ".", regex=False),
            errors="coerce"
        ) / 100
    )

    df["taux_chomage"] = df["nbr_chomeur_15_64"] / df["nbr_personnes_active_15_64"]

    df["densite"] = df["population"] / df["superficie"]


    return df


def remove_outliers(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Supprime les valeurs aberrantes selon des bornes définies par colonne.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à filtrer
    filters : dict
        Dictionnaire {colonne: (min, max)} définissant les bornes acceptables

    Returns
    -------
    pd.DataFrame
        DataFrame filtré

    """
    df = df.copy()
    before = len(df)

    for col, (lower, upper) in filters.items():
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    after = len(df)
    print(f"Outliers supprimés : {before - after} lignes ({(before - after) / before * 100:.1f}%)")

    return df


def merge_all(dvf: pd.DataFrame, insee: pd.DataFrame, log_soc: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne les trois sources de données sur le code commune.

    Parameters
    ----------
    dvf : pd.DataFrame
        DataFrame DVF nettoyé
    insee : pd.DataFrame
        DataFrame INSEE nettoyé
    log_soc : pd.DataFrame
        DataFrame des logements sociaux

    Returns
    -------
    pd.DataFrame
        DataFrame fusionné DVF + INSEE + logements sociaux
    """
    dvf = dvf.copy()
    insee = insee.copy()
    log_soc = log_soc.copy()

    # Harmoniser les types de la clé de jointure
    dvf["code_commune"] = dvf["code_commune"].astype("string")
    insee["code_commune"] = insee["code_commune"].astype("string")
    log_soc["code_commune"] = log_soc["code_commune"].astype("string")

    # Fusion
    df = pd.merge(dvf, insee, on="code_commune", how="left")
    df = pd.merge(df, log_soc, on="code_commune", how="left")

    return df
