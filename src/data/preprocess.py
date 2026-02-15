"""
Prétraitement des données brutes : nettoyage des dates et calcul de l'absence.
Aucune fuite de données : toutes les variables sont dérivées des colonnes brutes.
"""
import logging
import pandas as pd

logger = logging.getLogger("football_injury_risk")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et prépare le jeu de données brut.

    - Conversion des colonnes de date en datetime.
    - Suppression des lignes avec dates invalides.
    - Calcul de la durée d'absence en jours (absence_days).
    - Filtrage des absences incohérentes (<= 0).

    Parameters
    ----------
    df : pd.DataFrame
        Données brutes (format CSV inchangé).

    Returns
    -------
    pd.DataFrame
        Données nettoyées avec colonne absence_days.
    """
    df = df.copy()
    initial_rows = len(df)

    # Conversion des dates (errors='coerce' pour éviter les crashs sur formats invalides)
    df["Date of Injury"] = pd.to_datetime(df["Date of Injury"], errors="coerce")
    df["Date of return"] = pd.to_datetime(df["Date of return"], errors="coerce")

    # Suppression des lignes où les dates n'ont pas pu être converties
    df = df.dropna(subset=["Date of Injury", "Date of return"])
    dropped_dates = initial_rows - len(df)
    if dropped_dates > 0:
        logger.info("Dropped %d rows with invalid dates.", dropped_dates)

    # Durée d'absence en jours (sans fuite : uniquement dates de cette blessure)
    df["absence_days"] = (df["Date of return"] - df["Date of Injury"]).dt.days

    # Suppression des absences négatives ou nulles (incohérences)
    before = len(df)
    df = df[df["absence_days"] > 0]
    dropped_abs = before - len(df)
    if dropped_abs > 0:
        logger.info("Dropped %d rows with non-positive absence_days.", dropped_abs)

    logger.info("Preprocessing done: %d rows remaining.", len(df))
    return df.reset_index(drop=True)
