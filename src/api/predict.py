"""
Prediction logic: build feature vector from request and call model.predict_proba().
Uses config for feature names and risk thresholds; no ML pipeline code changed.
"""
import logging
import pandas as pd

from src.utils.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    RISK_LEVEL_LOW_MAX,
    RISK_LEVEL_MEDIUM_MAX,
)

logger = logging.getLogger("football_injury_risk")


def request_to_dataframe(age: int, position: str, injury: str, fifa_rating: float,
                        absence_days: int, previous_injuries: int,
                        days_since_last_injury: int) -> pd.DataFrame:
    """
    Build a single-row DataFrame with the same columns and order as the training pipeline.
    Derived features: Age_squared, serious_injury, risk_ratio.
    """
    # Use 999 for "first injury" if user sends a high value; cap for ratio
    days_since = days_since_last_injury if days_since_last_injury < 500 else 999
    serious_injury = 1 if absence_days > 60 else 0
    risk_ratio = absence_days / (days_since + 1)

    row = {
        "Age": age,
        "Age_squared": age ** 2,
        "FIFA rating": fifa_rating,
        "absence_days": absence_days,
        "serious_injury": serious_injury,
        "previous_injuries": previous_injuries,
        "days_since_last_injury": days_since,
        "risk_ratio": risk_ratio,
        "Position": position.strip(),
        "Injury": injury.strip(),
    }
    return pd.DataFrame([row])


def probability_to_risk_level(probability: float) -> str:
    """Map probability to Low / Medium / High using config thresholds."""
    if probability < RISK_LEVEL_LOW_MAX:
        return "Low"
    if probability < RISK_LEVEL_MEDIUM_MAX:
        return "Medium"
    return "High"


def run_predict(model, req) -> tuple[float, str]:
    """
    Run model.predict_proba on the request, return (injury_probability, risk_level).
    """
    df = request_to_dataframe(
        age=req.age,
        position=req.position,
        injury=req.injury,
        fifa_rating=req.fifa_rating,
        absence_days=req.absence_days,
        previous_injuries=req.previous_injuries,
        days_since_last_injury=req.days_since_last_injury,
    )
    # Ensure column order matches training (numeric + categorical)
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = df[[c for c in columns if c in df.columns]]
    try:
        proba = model.predict_proba(df)[0, 1]
    except Exception as e:
        logger.exception("predict_proba failed: %s", e)
        raise
    risk = probability_to_risk_level(proba)
    return float(proba), risk
