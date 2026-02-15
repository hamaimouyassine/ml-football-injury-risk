"""
FastAPI application: web interface for Football Injury Risk Prediction.
Model is loaded once at startup. No refactoring of existing ML pipeline.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.routes import router as api_router
from src.utils.config import BEST_MODEL_PATH, PROJECT_ROOT

logger = logging.getLogger("football_injury_risk")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; clean up on shutdown."""
    model_path = Path(BEST_MODEL_PATH)
    if not model_path.exists():
        logger.warning("Model file not found at %s. Run: python -m src.pipelines.pipeline", model_path)
        app.state.model = None
    else:
        try:
            app.state.model = joblib.load(model_path)
            logger.info("Model loaded from %s", model_path)
        except Exception as e:
            logger.exception("Failed to load model: %s", e)
            app.state.model = None
    yield
    app.state.model = None


app = FastAPI(
    title="Football Injury Risk Predictor",
    description="Predict reinjury probability within 90 days after return.",
    version="1.0.0",
    lifespan=lifespan,
)

# Static and template dirs (routes use Jinja2Templates with PROJECT_ROOT / "templates")
static_dir = PROJECT_ROOT / "static"
static_dir.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "templates").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.include_router(api_router, tags=["web"])


@app.get("/health")
async def health():
    """Health check for deployment (e.g. Render, Railway)."""
    return {"status": "ok"}
