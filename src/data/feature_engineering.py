"""
Feature engineering : cible (récidive), historique joueur, variables dérivées.
Toutes les features sont construites sans data leakage (pas d'info future).
"""
import logging
import pandas as pd

from src.utils.config import REINJURY_WINDOW_DAYS

logger = logging.getLogger("football_injury_risk")


def create_reinjury_target(df: pd.DataFrame, window_days: int = REINJURY_WINDOW_DAYS) -> pd.DataFrame:
    """
    Crée la cible binaire : récidive dans les N jours après le retour.

    Pour chaque blessure (sauf la dernière par joueur), on regarde si la
    prochaine blessure du même joueur survient entre 0 et window_days jours
    après la date de return. Si oui -> reinjury = 1, sinon 0.

    Parameters
    ----------
    df : pd.DataFrame
        Doit contenir Name, Date of Injury, Date of return.
    window_days : int
        Fenêtre en jours après le retour (défaut 90).

    Returns
    -------
    pd.DataFrame
        DataFrame avec colonne reinjury_90 (ou reinjury_{window_days}).
    """
    df = df.sort_values(["Name", "Date of Injury"]).copy()
    target_col = f"reinjury_{window_days}"
    df[target_col] = 0

    for player in df["Name"].unique():
        player_df = df[df["Name"] == player]
        for i in range(len(player_df) - 1):
            current_idx = player_df.index[i]
            next_idx = player_df.index[i + 1]
            return_date = df.loc[current_idx, "Date of return"]
            next_injury_date = df.loc[next_idx, "Date of Injury"]
            delta = (next_injury_date - return_date).days
            if 0 <= delta <= window_days:
                df.loc[current_idx, target_col] = 1

    logger.info("Target '%s' created. Positive rate: %.2f%%", target_col, 100 * df[target_col].mean())
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features d'historique par joueur (sans fuite : passé uniquement).

    - previous_injuries : nombre de blessures déjà subies par le joueur avant cette ligne.
    - days_since_last_injury : jours entre le return de la blessure précédente et cette blessure.
      Mis à -1 pour la première blessure du joueur (sera remplacé plus tard pour le modèle).
    """
    df = df.sort_values(["Name", "Date of Injury"]).copy()
    df["previous_injuries"] = 0
    df["days_since_last_injury"] = -1

    for player in df["Name"].unique():
        player_df = df[df["Name"] == player]
        for i in range(len(player_df)):
            idx = player_df.index[i]
            df.loc[idx, "previous_injuries"] = i
            if i > 0:
                prev_idx = player_df.index[i - 1]
                prev_return = df.loc[prev_idx, "Date of return"]
                current_injury = df.loc[idx, "Date of Injury"]
                delta = (current_injury - prev_return).days
                df.loc[idx, "days_since_last_injury"] = delta

    return df


def add_severity_feature(df: pd.DataFrame, threshold_days: int = 60) -> pd.DataFrame:
    """Blessure considérée comme grave si absence_days > threshold_days."""
    df = df.copy()
    df["serious_injury"] = (df["absence_days"] > threshold_days).astype(int)
    return df


def add_risk_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio absence_days / (days_since_last_injury + 1).
    Pour la première blessure (days_since = -1), on utilise 999 pour éviter division par zéro.
    """
    df = df.copy()
    days_since = df["days_since_last_injury"].replace(-1, 999)
    df["risk_ratio"] = df["absence_days"] / (days_since + 1)
    return df


def add_age_squared(df: pd.DataFrame) -> pd.DataFrame:
    """Terme non linéaire pour l'âge (risque peut varier de façon non linéaire)."""
    df = df.copy()
    df["Age_squared"] = df["Age"] ** 2
    return df


def prepare_model_dataset(
    df: pd.DataFrame,
    target_col: str = "reinjury_90",
) -> pd.DataFrame:
    """
    Applique tout le feature engineering et retourne le jeu prêt pour le modèle.

    - Remplace days_since_last_injury = -1 par 999 (première blessure = "très loin").
    - Garde uniquement les colonnes utilisées par le pipeline (numériques + catégorielles + cible).
    """
    df = df.copy()
    df["days_since_last_injury"] = df["days_since_last_injury"].replace(-1, 999)

    feature_cols = [
        "Age",
        "Age_squared",
        "FIFA rating",
        "absence_days",
        "serious_injury",
        "previous_injuries",
        "days_since_last_injury",
        "risk_ratio",
        "Position",
        "Injury",
        target_col,
    ]
    # Keep only columns that exist (e.g. if target_col has different suffix)
    available = [c for c in feature_cols if c in df.columns]
    df = df[available]
    logger.info("Model dataset prepared: shape %s, columns %s", df.shape, list(df.columns))
    return df


def run_full_feature_engineering(
    df: pd.DataFrame,
    window_days: int = REINJURY_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Enchaîne toutes les étapes de feature engineering dans le bon ordre.
    À appeler après preprocess_data().
    """
    df = create_reinjury_target(df, window_days=window_days)
    df = add_history_features(df)
    df = add_severity_feature(df)
    df = add_risk_ratio(df)
    df = add_age_squared(df)
    df = prepare_model_dataset(df, target_col=f"reinjury_{window_days}")
    return df
