"""
Exemple : charger le meilleur modèle et faire des prédictions (ou évaluer sur les données).
Lancer depuis la racine du projet : python scripts/predict.py
"""
import sys
from pathlib import Path

# Ajouter la racine du projet au path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.utils.config import BEST_MODEL_PATH, RAW_DATA_PATH
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.data.feature_engineering import run_full_feature_engineering


def load_model():
    """Charge le modèle sauvegardé (pipeline preprocessor + classifier)."""
    path = Path(BEST_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle non trouvé : {path}. Lance d'abord : python -m src.pipelines.pipeline"
        )
    return joblib.load(path)


def get_test_data():
    """Charge et prépare les données comme à l'entraînement (sans fuite)."""
    df = load_raw_data(str(RAW_DATA_PATH))
    df = preprocess_data(df)
    df = run_full_feature_engineering(df)
    return df


def main():
    print("Chargement du modèle...")
    model = load_model()

    print("Chargement des données (prétraitement + features)...")
    df = get_test_data()

    X = df.drop(columns=["reinjury_90"])
    y_true = df["reinjury_90"]

    # Prédictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Métriques rapides
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    print("\n--- Résultats sur tout le jeu préparé (à titre indicatif) ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 weighted : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"ROC-AUC : {roc_auc_score(y_true, y_prob):.4f}")

    # Exemple : prédiction pour les 3 premières lignes
    print("\n--- Exemple : 3 premières lignes ---")
    sample = X.head(3)
    pred = model.predict(sample)
    prob = model.predict_proba(sample)[:, 1]
    for i in range(len(sample)):
        print(f"  Ligne {i+1} : prédit={pred[i]}, proba récidive={prob[i]:.3f}")


if __name__ == "__main__":
    main()
