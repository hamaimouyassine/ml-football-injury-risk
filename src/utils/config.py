"""
Configuration centralisée du projet Football Injury Risk.
Tous les chemins, constantes et hyperparamètres sont définis ici.
"""
import logging
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RAW_DATA_PATH = DATA_RAW_DIR / "player_injuries_impact.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# API: risk level thresholds (probability -> Low / Medium / High)
RISK_LEVEL_LOW_MAX = 0.3
RISK_LEVEL_MEDIUM_MAX = 0.6

# Calibration: "sigmoid" (Platt), "isotonic", or None to disable
# Calibration can spread probabilities and improve interpretability
CALIBRATION_METHOD = None  # ancienne calibration (sans CalibratedClassifierCV)
CALIBRATION_CV = 5

# Web app
APP_VERSION = "1.0.0"

# -----------------------------------------------------------------------------
# Data & target
# -----------------------------------------------------------------------------
TARGET_COLUMN = "reinjury_90"
REINJURY_WINDOW_DAYS = 90

# Features used in the ML pipeline (must match feature_engineering output)
NUMERIC_FEATURES = [
    "Age",
    "Age_squared",
    "FIFA rating",
    "absence_days",
    "serious_injury",
    "previous_injuries",
    "days_since_last_injury",
    "risk_ratio",
]
CATEGORICAL_FEATURES = ["Position", "Injury"]

# -----------------------------------------------------------------------------
# Train / validation
# -----------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# -----------------------------------------------------------------------------
# Model hyperparameters (defaults)
# -----------------------------------------------------------------------------
LOGISTIC_REGRESSION_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_split": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "verbosity": -1,
    "force_col_wise": True,
}

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return project logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("football_injury_risk")
