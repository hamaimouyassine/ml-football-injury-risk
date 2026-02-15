"""
Web and API routes: home, prediction form, POST /predict, about.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.api.schemas import PredictRequest, PredictResponse
from src.api.predict import run_predict
from src.utils.config import PROJECT_ROOT, METRICS_PATH, APP_VERSION

logger = logging.getLogger("football_injury_risk")

BUILD_DATE = datetime.utcnow().strftime("%Y-%m-%d")


def get_top_feature_importance(metrics: dict | None, top_n: int = 8) -> list[tuple[str, float]]:
    """Top N features by importance for About bar chart."""
    if not metrics:
        return []
    fi = metrics.get("feature_importance") or {}
    items = []
    for k, v in fi.items():
        if not isinstance(v, (int, float)):
            continue
        v = float(v)
        label = k.replace("Position_", "").replace("Injury_", "")
        if len(label) > 22:
            label = label[:19] + "..."
        items.append((label, v))
    items.sort(key=lambda x: -x[1])
    return items[:top_n]

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

router = APIRouter()

# Positions: exact match with training data (one has trailing space)
POSITIONS = [
    "Attacking Midfielder", "Center Back", "Center Forward", "Central Midfielder",
    "Central Midfielder ", "Defensive Midfielder", "Defensive Midfielder ",
    "Goalkeeper", "Left Back", "Left Midfielder", "Left winger",
    "Right Back", "Right Midfielder", "Right winger",
]


def get_injuries_list() -> list[str]:
    """Injury types: from metrics feature_importance (model-known) + Unknown injury fallback."""
    injuries_set = {"Unknown injury"}
    path = Path(METRICS_PATH)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            fi = data.get("feature_importance") or {}
            for key in fi:
                if key.startswith("Injury_"):
                    injuries_set.add(key.replace("Injury_", "", 1))
        except Exception:
            pass
    # Fallback if no metrics
    if len(injuries_set) <= 1:
        injuries_set.update([
            "Hamstring strain", "Ankle injury", "Knee injury", "Calf injury",
            "Groin injury", "Back injury", "Hip injury", "Foot injury",
            "Muscle strain", "Cruciate ligament tear", "Unknown injury",
        ])
    return sorted(injuries_set)


def get_metrics_for_about() -> dict | None:
    """Load saved metrics (ROC-AUC, etc.) for the About page."""
    path = Path(METRICS_PATH)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load metrics for About page: %s", e)
        return None


def _base_ctx(request: Request) -> dict:
    return {"request": request, "version": APP_VERSION, "build_date": BUILD_DATE}


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page: title, short explanation, Start Prediction button."""
    return templates.TemplateResponse("home.html", _base_ctx(request))


@router.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Prediction page: form and result area (result filled via JS or server-side after POST)."""
    ctx = _base_ctx(request)
    ctx.update({"positions": POSITIONS, "injuries": get_injuries_list()})
    return templates.TemplateResponse("predict.html", ctx)


@router.post("/predict", response_model=PredictResponse)
async def predict_api(request: Request, body: PredictRequest):
    """
    API: accept JSON with features, return injury_probability and risk_level.
    Model loaded at startup in app.state.model.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline first.")
    try:
        prob, risk = run_predict(model, body)
        return PredictResponse(injury_probability=prob, risk_level=risk)
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed.")


@router.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """About page: problem, model, metrics, limitations, disclaimer."""
    metrics = get_metrics_for_about()
    ctx = _base_ctx(request)
    ctx.update({"metrics": metrics, "top_features": get_top_feature_importance(metrics)})
    return templates.TemplateResponse("about.html", ctx)
