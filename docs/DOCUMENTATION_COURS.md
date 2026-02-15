# Documentation cours – Football Injury Risk Prediction

**Projet :** ML Football Injury Risk (production-ready)  
**Objectif :** Comprendre le projet, le pipeline ML et être prêt pour toute question d’entretien (technique, Master 2 IA, portfolio).

---

## 1. Vue d’ensemble du projet

### 1.1 Objectif métier

Prédire **la probabilité qu’un joueur professionnel se blesse à nouveau dans une fenêtre de temps donnée** (par défaut 90 jours après son retour).  
C’est un problème de **classification binaire** : récidive dans les 90 jours (1) ou non (0).

### 1.2 Stack technique

| Composant        | Technologie    |
|------------------|----------------|
| Langage          | Python         |
| Données          | Pandas         |
| ML / préparation | Scikit-learn   |
| Modèles          | Logistic Regression, Random Forest, **LightGBM** |
| Persistance      | Joblib (meilleur modèle + métriques JSON) |
| Logging          | `logging` standard |

### 1.3 Structure du dépôt (production-ready)

```
ml-football-injury-risk/
├── data/
│   └── raw/
│       └── player_injuries_impact.csv
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── pipelines/
│   │   └── pipeline.py
│   └── utils/
│       └── config.py
├── notebooks/
├── models/                    # best_model.joblib, metrics.json (générés)
├── docs/
│   └── DOCUMENTATION_COURS.md
├── requirements.txt
└── README.md
```

---

## 2. Les données

### 2.1 Source

- **Fichier :** `data/raw/player_injuries_impact.csv` (format inchangé).
- **Contexte :** blessures en football professionnel européen.
- **Granularité :** une ligne = une blessure (un joueur peut avoir plusieurs lignes).

### 2.2 Colonnes principales (brutes)

| Colonne           | Type / rôle |
|-------------------|-------------|
| `Name`            | Identifiant joueur |
| `Position`        | Poste (Center Back, Left Back, etc.) |
| `Age`, `FIFA rating` | Numériques |
| `Injury`          | Type de blessure |
| `Date of Injury`, `Date of return` | Dates |
| Colonnes Match*   | Optionnel pour le modèle actuel |

### 2.3 Après chargement

- Conversion des dates en `datetime`, suppression des lignes invalides.
- **absence_days** = Date of return - Date of Injury (jours).
- Filtrage des absences ≤ 0.

---

## 3. Prétraitement (`src/data/preprocess.py`)

- **Conversion des dates** avec `errors="coerce"` pour éviter les crashs.
- **Suppression** des lignes avec dates ou absence incohérentes.
- **Aucune fuite** : tout est dérivé des colonnes brutes de la ligne.

---

## 4. Feature engineering (`src/data/feature_engineering.py`)

### 4.1 Cible : `reinjury_90`

- **create_reinjury_target(df, window_days=90)** : pour chaque blessure (sauf la dernière par joueur), on regarde si la **prochaine** blessure du même joueur survient entre 0 et 90 jours après le return. Si oui → 1, sinon 0.

### 4.2 Historique (par joueur)

- **previous_injuries** : nombre de blessures déjà subies avant cette ligne.
- **days_since_last_injury** : jours entre le return de la blessure précédente et cette blessure (-1 pour la première, remplacé par 999 pour le modèle).

### 4.3 Dérivées

- **serious_injury** : 1 si absence_days > 60, sinon 0.
- **risk_ratio** : absence_days / (days_since_last_injury + 1).
- **Age_squared** : Age² (terme non linéaire).

### 4.4 Préparation pour le modèle

- **run_full_feature_engineering(df)** enchaîne tout dans le bon ordre.
- **prepare_model_dataset** garde uniquement les colonnes utilisées par le pipeline (définies dans `config.py` : NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN).

---

## 5. Modèles et entraînement (`src/models/train.py`)

### 5.1 Configuration (`src/utils/config.py`)

- **TARGET_COLUMN** = "reinjury_90".
- **NUMERIC_FEATURES** : Age, Age_squared, FIFA rating, absence_days, serious_injury, previous_injuries, days_since_last_injury, risk_ratio.
- **CATEGORICAL_FEATURES** : Position, Injury.
- **TEST_SIZE** = 0.2, **RANDOM_STATE** = 42, **CV_FOLDS** = 5.

### 5.2 Pipeline sklearn

- **ColumnTransformer** :
  - **num** : StandardScaler sur les colonnes numériques.
  - **cat** : OneHotEncoder(handle_unknown="ignore") sur Position et Injury.
  - **remainder="drop"** : aucune autre colonne (pas de fuite).
- **Classifieur** : LogisticRegression, RandomForestClassifier ou LGBMClassifier.

### 5.3 Split et validation

- **Train/test** : 80/20, **stratify=y** (même proportion de récidives).
- **Cross-validation** : 5-fold **StratifiedKFold** sur le train (accuracy, F1 weighted, ROC-AUC).

### 5.4 Les trois modèles

| Modèle               | Rôle / propriété |
|----------------------|-------------------|
| **Logistic Regression** | Baseline, interprétable (coefficients). |
| **Random Forest**    | Non linéaire, robuste, feature_importances_. |
| **LightGBM**         | Gradient boosting, souvent meilleur ROC-AUC. |

### 5.5 Feature importance

- **LogisticRegression** : valeur absolue des coefficients.
- **Random Forest / LightGBM** : feature_importances_.
- Les noms des features après transformation (num + one-hot cat) sont récupérés depuis le ColumnTransformer pour afficher l’importance.

### 5.6 Sauvegarde du meilleur modèle

- **run_all_models(df)** entraîne les 3 modèles ; après chaque entraînement, si le **ROC-AUC test** est supérieur au ROC-AUC déjà sauvegardé dans `models/metrics.json`, on écrase **models/best_model.joblib** et **models/metrics.json**.
- Métriques sauvegardées : model_name, roc_auc, accuracy, f1_weighted, cv_metrics, feature_importance.

---

## 6. Évaluation (`src/models/evaluate.py`)

- **evaluate_classification(y_true, y_pred, y_prob)** retourne : **accuracy**, **f1_macro**, **f1_weighted**, **roc_auc** (si y_prob fourni).
- **evaluate_regression** (prévu pour extension) : RMSE, MAE, R².
- **save_metrics(metrics, path)** : sauvegarde en JSON (valeurs converties en types sérialisables).

---

## 7. Pipeline complet (`src/pipelines/pipeline.py`)

- **run_full_pipeline(data_path=None, run_all=True, save_best=True)** :
  1. load_raw_data → preprocess_data → run_full_feature_engineering.
  2. Si run_all : run_all_models(df) (3 modèles, sauvegarde du meilleur).
  3. Sinon : train_and_evaluate(df, "logistic_regression", save_best).

**Lancer tout :**
```bash
python -m src.pipelines.pipeline
```

---

## 8. Métriques utilisées

| Métrique      | Rôle |
|---------------|------|
| **Accuracy**  | % de prédictions correctes (peut être trompeuse en déséquilibré). |
| **F1 (macro)** | Moyenne des F1 par classe (équilibre précision/rappel). |
| **F1 (weighted)** | F1 pondéré par la taille des classes. |
| **ROC-AUC**   | Capacité à séparer les classes (probabilités) ; 0.5 = hasard, 1 = parfait. |

Pour la **récidive** (classe souvent minoritaire) : le **rappel** sur la classe 1 indique combien de récidives on détecte ; le **F1** sur la classe 1 résume le compromis précision/rappel.

---

## 9. FAQ Machine Learning – Prêt pour l’entretien

### 9.1 Projet et objectif

- **Quel est le problème ?** Classification binaire : récidive dans les 90 jours après retour (oui/non).
- **Pourquoi 90 jours ?** Fenêtre métier pour la prévention ; configurable via `REINJURY_WINDOW_DAYS`.
- **Supervisé ou non ?** Supervisé (cible binaire construite à partir des dates).

### 9.2 Données et fuite

- **Data leakage ?** On n’utilise que des infos **passées** au moment de la blessure (dates, historique du joueur). Pas d’info future (ex. date de la prochaine blessure) dans les features.
- **Déséquilibre ?** Souvent oui (moins de récidives) → `class_weight="balanced"` et métriques F1/ROC-AUC.

### 9.3 Features

- **Variables temporelles** : days_since_last_injury, absence_days, previous_injuries → très liées au risque de récidive.
- **Position / Injury** : catégorielles → One-Hot (pas d’ordre arbitraire).
- **StandardScaler** : pour les numériques ; nécessaire pour la régression logistique (optimisation) et utile pour la stabilité.

### 9.4 Modèles

- **Pourquoi 3 modèles ?** Baseline (logistic), modèles plus puissants (RF, LightGBM) pour comparer et choisir le meilleur (ici par ROC-AUC).
- **Pipeline avec préprocessing ?** Reproductibilité et **même préprocessing à l’inférence** (scaler + encoder) sans refaire le code à la main.
- **Régularisation (logistic) ?** On peut ajouter penalty="l2", C=... pour limiter le surajustement.

### 9.5 Validation et métriques

- **Pourquoi cross-validation ?** Estimer la performance de façon plus stable qu’un seul split ; StratifiedKFold pour garder les proportions de classes.
- **Métrique principale ?** ROC-AUC pour la qualité des scores ; F1/rappel sur la classe 1 pour l’usage métier (détection des récidives).
- **Pourquoi ne pas optimiser que l’accuracy ?** En déséquilibré, un modèle “toujours 0” peut avoir une bonne accuracy mais un mauvais rappel sur la classe 1.

### 9.6 Production et déploiement

- **Chargement du modèle ?** `joblib.load("models/best_model.joblib")` ; le pipeline (preprocessor + classifier) est sauvegardé en un seul objet.
- **À l’inférence ?** Passer les **mêmes colonnes** (ordre et noms) que à l’entraînement ; le ColumnTransformer applique scale + one-hot.

### 9.7 Concepts ML généraux (entretien)

- **Bias / variance ?** Modèle trop simple (bias élevé) vs trop complexe (variance élevée, overfitting). Random Forest / LightGBM : max_depth, min_samples_split limitent la complexité.
- **Overfitting ?** Performance train >> test ; solutions : plus de données, régularisation, cross-validation, simplifier le modèle.
- **Régression vs classification ?** Ici c’est classification (réponse binaire). Si la cible était **absence_days** (nombre de jours), ce serait une régression (métriques : RMSE, MAE, R²).
- **Rappel vs précision ?** Rappel = parmi les vrais positifs, combien on en détecte. Précision = parmi les prédits positifs, combien sont vrais. En récidive, on peut privilégier le rappel (ne pas rater de récidives) au prix de plus de faux positifs.

### 9.8 Améliorations possibles

- GridSearch / RandomizedSearch pour les hyperparamètres.
- Courbes ROC et Precision-Recall pour visualiser le compromis selon le seuil.
- Gestion explicite des valeurs manquantes (imputation).
- Feature selection (SelectKBest, ou importance puis seuil).
- API (FastAPI) pour exposer une prédiction (probabilité de récidive).

---

## 10. Résumé en une page

| Étape        | Fichier / fonction                | Rôle principal |
|-------------|------------------------------------|----------------|
| Config      | `utils/config.py`                  | Chemins, features, hyperparamètres, logging |
| Données     | `data/load_data.load_raw_data`     | Charger le CSV |
| Nettoyage   | `data/preprocess.preprocess_data`  | Dates + absence_days |
| Features    | `data/feature_engineering`         | Cible, historique, dérivées, prepare_model_dataset |
| Entraînement| `models/train.train_and_evaluate`   | Split, CV, fit, métriques, importance, save best |
| Évaluation  | `models/evaluate.evaluate_classification` | Accuracy, F1, ROC-AUC |
| Pipeline    | `pipelines/pipeline.run_full_pipeline` | Enchaînement complet |

**Problème :** classification binaire (récidive dans les 90 jours).  
**Modèles :** Logistic Regression (baseline), Random Forest, LightGBM.  
**Métriques :** Accuracy, F1 (macro, weighted), ROC-AUC.  
**Production :** meilleur modèle (par ROC-AUC) sauvegardé en joblib ; métriques en JSON.

Tu peux utiliser ce document comme **base de révision** avant un entretien : structure du projet, choix de modèles et de métriques, absence de fuite de données, cross-validation, et réponses aux questions classiques (bias/variance, rappel/précision, déploiement).
