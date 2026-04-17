""" Collecte de données via url ou api"""
import pandas as pd
import requests
import zipfile
from io import BytesIO, TextIOWrapper


url_dvf = "https://www.data.gouv.fr/api/1/datasets/r/99a26050-b94f-4ffc-9eb0-73ed28a895d1"

api_dvf = "https://valoris-immo.fr/api/v1/prix-median"

url_insee = "depend de la base"

api_insee = "depend de la base"


def load_dvf_data_url(url: str) -> pd.DataFrame:
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

    Process
    -------
    1. Téléchargement du fichier depuis l'URL
    2. Décompression du fichier ZIP en mémoire
    3. Extraction du fichier texte contenu dans le ZIP
    4. Lecture du fichier en DataFrame pandas
    """
    # 1. download
    response = requests.get(url)
    response.raise_for_status()

    # 2. unzip
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        file_name = z.namelist()[0]  # premier fichier dans le zip

        with z.open(file_name) as f:
            # 3. lecture texte avec bon encodage
            df = pd.read_csv(
                f, encoding="utf-8",
                sep="|",
                low_memory=False
            )

    return df


"""
df = load_dvf_data_url(url_dvf)

print(df.head(3))
print(df.shape)
"""


def load_dvf_zip_csv(url: str) -> pd.DataFrame:
    """
    Télécharge et charge un fichier au format .zip contenant un CSV.

    Le fichier est compressé et contient un CSV structuré.

    Parameters
    ----------
    url : str
        URL pointant vers un fichier compressé (.zip contenant un CSV)

    Returns
    -------
    pd.DataFrame
        DataFrame pandas contenant les données DVF décompressées.

    Process
    -------
    1. Téléchargement du fichier ZIP depuis l'URL
    2. Ouverture du ZIP en mémoire (sans extraction disque)
    3. Lecture du fichier CSV contenu dans l'archive
    4. Retour du DataFrame pandas
    """

    # 1. Téléchargement du fichier ZIP
    response = requests.get(url)
    response.raise_for_status()

    # 2. Ouverture du ZIP en mémoire
    with zipfile.ZipFile(BytesIO(response.content)) as z:

        # récupération du premier fichier dans l’archive
        file_name = z.namelist()[0]

        # 3. Lecture du CSV
        with z.open(file_name) as f:

            df = pd.read_csv(
                TextIOWrapper(f, encoding="utf-8"),  # adapter en "latin-1" si besoin
                sep=";",                              # CSV ;
                low_memory=False
            )

    # 4. Retour du DataFrame
    return df
