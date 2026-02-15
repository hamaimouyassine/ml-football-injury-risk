"""
Pipeline ML complet : chargement -> prétraitement -> feature engineering -> entraînement.
Aucune fuite de données ; split train/test après toute la préparation.
"""
import logging
from pathlib import Path

import pandas as pd

from src.utils.config import RAW_DATA_PATH, setup_logging
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.data.feature_engineering import run_full_feature_engineering
from src.models.train import train_and_evaluate, run_all_models

logger = setup_logging()


def run_full_pipeline(
    data_path: str | Path | None = None,
    run_all: bool = True,
    save_best: bool = True,
) -> pd.DataFrame:
    """
    Exécute le pipeline complet.

    1. Chargement des données brutes (format CSV inchangé).
    2. Prétraitement (dates, absence_days).
    3. Feature engineering (cible, historique, dérivées, sélection).
    4. Entraînement : si run_all=True, entraîne les 3 modèles et sauvegarde le meilleur ;
       sinon entraîne uniquement la régression logistique.

    Parameters
    ----------
    data_path : str or Path, optional
        Chemin vers le CSV. Default : RAW_DATA_PATH du config.
    run_all : bool
        Si True, entraîne Logistic Regression, Random Forest et LightGBM.
    save_best : bool
        Si True, sauvegarde le meilleur modèle (joblib) et les métriques (JSON).

    Returns
    -------
    pd.DataFrame
        Jeu de données prêt pour le modèle (après feature engineering).
    """
    path = data_path or RAW_DATA_PATH
    logger.info("Starting full pipeline with data: %s", path)

    # 1. Load
    df = load_raw_data(str(path))

    # 2. Preprocess (no leakage)
    df = preprocess_data(df)

    # 3. Feature engineering (target + history + derived; no leakage)
    df = run_full_feature_engineering(df)

    # 4. Train / evaluate
    if run_all:
        run_all_models(df)
    else:
        train_and_evaluate(df, model_name="logistic_regression", save_best=save_best)

    logger.info("Pipeline finished.")
    return df


if __name__ == "__main__":
    run_full_pipeline(run_all=True, save_best=True)
