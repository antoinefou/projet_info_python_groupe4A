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
    
    # rename
    df_filtered = df_filtered.rename(columns={
        "Valeur fonciere": "valeur_fonciere",
        "Code postal": "code_postal",
        "Commune": "commune",
        "Code departement": "code_departement",
        "Code commune": "code_commune",
        "Type local": "type_local",
        "Surface reelle bati": "surface_reelle_bati",
        "Nature mutation": "nature_mutation"
    })

    df_filtered["code_departement"] = df_filtered["code_departement"].astype(str).str.zfill(2)
    df_filtered["code_commune"] = df_filtered["code_commune"].astype(str).str.zfill(3)

    df_filtered["code_commune"] = df_filtered["code_departement"] + df_filtered["code_commune"]

    return df_filtered


"""dvf = filter_dvf_columns(load__data_url_zip_txt(url_dvf))
print(dvf.dtypes)
print(dvf)"""


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
        "P22_ACT1564": "nbr_personnes_active_15_64"
    })

    # 2. Conversion des types
    df["code_commune"] = df["code_commune"].astype("string")
    df["mediane_niveau_vie"] = df["mediane_niveau_vie"].astype("string")

    df = df.dropna()

    return df


def merge_dvf_insee(dvf: pd.DataFrame, insee: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    """
    Effectue une jointure entre les données DVF et INSEE sur la variable 'code_commune'.

    Paramètres
    ----------
    dvf : pd.DataFrame
        DataFrame DVF (transactions immobilières)
    insee : pd.DataFrame
        DataFrame INSEE (variables socio-économiques)
    how : str, optional (default="left")
        Type de jointure ('left', 'inner', 'right', 'outer')

    Retour
    ------
    pd.DataFrame
        DataFrame fusionné
    """

    dvf = dvf.copy()
    insee = insee.copy()

    # Sécurisation des types (clé de jointure)
    dvf["code_commune"] = dvf["code_commune"].astype("string")
    insee["code_commune"] = insee["code_commune"].astype("string")

    # Jointure
    df_merged = pd.merge(
        dvf,
        insee,
        on="code_commune",
        how=how
    )

    return df_merged

# test
"""
from collect import url_dvf, url_insee, load__data_url_zip_txt, load_insee_dossier_complet

dvf = filter_dvf_columns(load__data_url_zip_txt(url_dvf))
insee = preprocess_insee(load_insee_dossier_complet(url_insee))

dvf = compute_prix_m2(dvf)

data_final = merge_dvf_insee(dvf, insee)
print(data_final)"""
