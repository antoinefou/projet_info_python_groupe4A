""" Collecte de données via url ou api"""
import pandas as pd
import requests
import json
import zipfile
from io import BytesIO


def load__data_url_zip_txt(url: str) -> pd.DataFrame:
    """
    Télécharge et charge un fichier DVF au format .txt.zip depuis une URL data.gouv.fr.

    Le fichier DVF contient des données de transactions immobilières en France.
    Il est compressé en ZIP et structuré sous forme de fichier texte séparé par '|'.

    Parameters
    ----------
    url : str
        URL pointant vers un fichier DVF compressé (.txt.zip)

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les données DVF décompressées et chargées.

    """

    df = pd.read_csv(url, sep="|", compression="zip", low_memory=False)

    return df



def load_insee_dossier_complet(url: str) -> pd.DataFrame:
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as z:
        file_name = [f for f in z.namelist() if f.endswith(".csv") and "meta" not in f][0]

        with z.open(file_name) as f:
            df = pd.read_csv(
                f, encoding="utf-8",
                sep=";",                # INSEE utilise ; pas |
                low_memory=False,
                usecols=["CODGEO", "MED21", "TP6021", "P22_POP", "P22_CHOM1564", "P22_ACT1564", "SUPERF"]
            )

    return df



def load_logements_sociaux(url: str) -> pd.DataFrame:
    """
    Charge le taux de logements sociaux par commune depuis data.gouv.fr.

    Parameters
    ----------
    url : str
        URL du fichier CSV

    Returns
    -------
    pd.DataFrame
        DataFrame avec code_commune et taux_logements_sociaux
    """
    df = pd.read_csv(url, sep=";", low_memory=False)
    df = df[["Code Commune", "Taux de logements sociaux (%)"]].copy()
    df.columns = ["code_commune", "taux_logements_sociaux"]
    df["code_commune"] = df["code_commune"].astype("string")
    return df
