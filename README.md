# Football Injury Risk Prediction

**Production-ready ML project** to predict reinjury probability in professional football (classification).  
Suitable for technical interviews, Master 2 AI applications, and portfolio (Kaggle-level).

---

## Problem

- **Goal:** Predict whether a player will sustain a **reinjury within 90 days** after returning from a previous injury.
- **Target:** Binary classification (`reinjury_90`: 0 = no, 1 = yes).
- **Data:** European professional football injuries (one row per injury; multiple rows per player over time).

---

## Project structure

```
ml-football-injury-risk/
├── data/
│   └── raw/
│       └── player_injuries_impact.csv   # Raw data (format unchanged)
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI app, model load at startup
│   │   ├── routes.py         # Web + POST /predict
│   │   ├── schemas.py        # Pydantic request/response
│   │   └── predict.py        # Prediction logic (no ML pipeline change)
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
├── templates/                # Jinja2 (home, predict, about)
├── static/
├── notebooks/
├── models/                    # Saved best model + metrics (generated)
├── docs/
│   └── DOCUMENTATION_COURS.md
├── requirements.txt
└── README.md
```

---

## Data preprocessing

1. **Load** raw CSV (no format change).
2. **Dates:** Convert `Date of Injury` and `Date of return` to `datetime`; drop rows with invalid dates.
3. **Absence:** `absence_days = Date of return - Date of Injury` (days).
4. **Filter:** Remove rows with `absence_days <= 0`.

No future information is used (no data leakage).

---

## Feature engineering

All features are built from **past information only** (no leakage):

| Feature | Description |
|--------|-------------|
| **Target** | `reinjury_90`: 1 if next injury occurs within 90 days after return, else 0 |
| `previous_injuries` | Number of prior injuries for that player |
| `days_since_last_injury` | Days between previous return and this injury (999 if first injury) |
| `absence_days` | Duration of this injury (days) |
| `serious_injury` | 1 if `absence_days > 60`, else 0 |
| `risk_ratio` | `absence_days / (days_since_last_injury + 1)` |
| `Age_squared` | Age² (non-linear term) |
| `Age`, `FIFA rating` | Raw (or from data) |
| `Position`, `Injury` | Categorical (one-hot encoded in pipeline) |

---

## Models

- **Logistic Regression** (baseline, interpretable).
- **Random Forest** (non-linear, robust).
- **LightGBM** (gradient boosting, often best performance).

**Pipeline (sklearn):**

- `ColumnTransformer`: numeric features → `StandardScaler`; categorical → `OneHotEncoder(handle_unknown="ignore")`.
- Classifier fitted on the transformed data.
- Train/test split: 80/20, stratified, `random_state=42`.
- **5-fold stratified cross-validation** on the training set for stability.

---

## Metrics

- **Accuracy**
- **F1 (macro and weighted)**
- **ROC-AUC**

Feature importance is computed (coefficients for Logistic Regression, `feature_importances_` for tree-based models) and logged.

---

## How to run

### 1. Environment

```bash
pip install -r requirements.txt
```

### 2. Data

Place your raw CSV at:

```
data/raw/player_injuries_impact.csv
```

(Column names and format must match the expected schema; see Data preprocessing above.)

### 3. Full pipeline (train all models and save the best)

```bash
python -m src.pipelines.pipeline
```

Or from Python:

```python
from src.pipelines.pipeline import run_full_pipeline

# Trains Logistic Regression, Random Forest, LightGBM; saves best model by ROC-AUC
df = run_full_pipeline(run_all=True, save_best=True)
```

### 4. Single model (e.g. baseline only)

```python
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.data.feature_engineering import run_full_feature_engineering
from src.models.train import train_and_evaluate
from src.utils.config import RAW_DATA_PATH

df = load_raw_data(str(RAW_DATA_PATH))
df = preprocess_data(df)
df = run_full_feature_engineering(df)
model, results = train_and_evaluate(df, model_name="logistic_regression", save_best=True)
```

### 5. Load the saved best model

```python
import joblib
from pathlib import Path

model = joblib.load(Path("models/best_model.joblib"))
# Predict: model.predict(X) or model.predict_proba(X)
```

Metrics (and feature importance) are saved in `models/metrics.json`.

---

## Web interface (FastAPI)

A production-ready web UI is provided to run predictions in the browser.

### Run the backend

1. **Train and save the model** (if not already done):

   ```bash
   python -m src.pipelines.pipeline
   ```

2. **Start the API server**:

   ```bash
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

   - `--reload`: auto-reload on code change (dev only).
   - `--host 0.0.0.0`: listen on all interfaces (e.g. for Docker or cloud).
   - `--port 8000`: default port.

### Access the interface

- **In browser:** open [http://127.0.0.1:8000](http://127.0.0.1:8000).
- **Pages:** Home, **Predict** (form + result), **About** (problem, model, metrics, limitations, disclaimer).
- **API:** `POST /predict` with JSON body (see Pydantic schema in `src/api/schemas.py`). Response: `{ "injury_probability": float, "risk_level": "Low"|"Medium"|"High" }`.
- **Health check:** `GET /health` for deployment (e.g. Render, Railway).

Paths and risk thresholds are read from `src/utils/config.py` (no hardcoded paths).

### Test case: highest risk profile

With the current dataset and model, the **maximum reinjury probability** observed is around **50%** (the model is calibrated and does not output 90%+). To test the highest risk output:

- In the web UI: click **"Try example: highest risk profile (~50%)"** on the Predict page, then **Predict**.
- Or send this JSON to `POST /predict`:

| Field | Value |
|-------|--------|
| age | 32 |
| position | Center Forward |
| injury | Hamstring strain |
| fifa_rating | 74 |
| absence_days | 120 |
| previous_injuries | 10 |
| days_since_last_injury | 5 |

Expected: **~50%** probability, **Medium** risk. For 90%+ you would need a larger dataset or different model calibration.

---

## Results

After running the pipeline, check:

- **Console:** Cross-validation and test metrics per model; feature importance.
- **models/metrics.json:** Best model name, ROC-AUC, accuracy, F1, CV metrics, feature importance.
- **models/best_model.joblib:** Best model (by test ROC-AUC) for inference.

---

## Constraints respected

- Dataset format unchanged.
- No existing features removed.
- Modular, clean code; config in `src/utils/config.py`.
- No data leakage; scaling and encoding inside sklearn `Pipeline` for safe inference.

---

## Documentation

See **docs/DOCUMENTATION_COURS.md** for a full course-style documentation and interview preparation (FAQ ML, metrics, pipeline details).
