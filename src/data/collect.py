""" Collecte de données via url ou api"""
import pandas as pd
import requests
import json
import zipfile
from io import BytesIO


url_dvf = "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20260405-002321/valeursfoncieres-2025.txt.zip"

api_dvf = "https://valoris-immo.fr/api/v1/prix-median"

url_insee = "depend de la base"

# densité de population
api_insee1 = "https://api.insee.fr/melodi/data/DS_ESTIMATION_POPULATION?TIME_PERIOD=2025&GEO=DEP&maxResult=100000"

# revenu
api_insee2 = "https://api.insee.fr/melodi/data/DS_BTS_SAL_EQTP_SEX_PCS?TIME_PERIOD=2023&maxResult=20000"


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


url_api = "https://api.insee.fr/melodi/data/DS_ESTIMATION_POPULATION?TIME_PERIOD=2025&GEO=DEP&maxResult=100000"


def load_data_api(api_url):
    get_data = requests.get(api_url, verify=False)
    data_from_net = get_data.content
    data = json.loads(data_from_net)

    # Extraction des observations du jeu de données filtré, sur lesquelles on va boucler
    observations = data['observations']
    extracted_data = []

    # Boucle de lecture des observations dans le json 
    for obs in observations:
        dimensions = obs['dimensions']

        # Suivant les jeux de données attributes est présent ou non
        if 'attributes' in obs:
            attributes = obs['attributes']
        else:
            attributes = None

        # Suivant les jeux de données value peut être absent
        if 'value' in obs['measures']['OBS_VALUE_NIVEAU']:
            measures = obs['measures']['OBS_VALUE_NIVEAU']['value']
        else:
            mesures = None

        # on rassemble tout dans un objet
        if 'attributes' in obs:
            combined_data = {**dimensions, **attributes, 'OBS_VALUE_NIVEAU': measures}
        else:
            combined_data = {**dimensions, 'OBS_VALUE_NIVEAU': measures}

        extracted_data.append(combined_data)

    # Création d'un dataframe python
    df = pd.DataFrame(extracted_data)
    return df
"""print(load_data_api(api_insee1))"""

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
                usecols=["CODGEO", "MED21", "TP6021", "P22_POP", "P22_CHOM1564", "P22_ACT1564"]
            )

    return df

url_insee = "https://www.insee.fr/fr/statistiques/fichier/5359146/dossier_complet.zip"
"""insee = load_insee_dossier_complet(url_insee)

# 1. Forme du tableau
insee.shape

# 2. Premières lignes
insee.head()

# 3. Types et valeurs manquantes
insee.info()

# 4. Stats descriptives
insee.describe()

# 5. Valeurs manquantes par colonne
insee.isnull().sum()


url_logements_sociaux = "https://www.data.gouv.fr/api/1/datasets/r/b0d30277-3a14-4673-a988-2fa6c11e030c"

logements_sociaux = pd.read_csv(url_logements_sociaux, sep=";", low_memory=False)

logements_sociaux = logements_sociaux[["Code Commune", "Taux de logements sociaux (%)"]].copy()
logements_sociaux.columns = ["CODGEO", "taux_logements_sociaux"]

print(logements_sociaux.shape)
print(logements_sociaux.columns.tolist())
logements_sociaux.head()"""