"""
Chargement des données brutes (format CSV inchangé).
"""
import logging
import pandas as pd

logger = logging.getLogger("football_injury_risk")


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Charge le fichier CSV brut.

    Parameters
    ----------
    path : str
        Chemin vers le CSV (ex. data/raw/player_injuries_impact.csv).

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path)
    logger.info("Loaded raw data from %s: shape %s", path, df.shape)
    return df
