"""FastAPI service exposing the trained heart-disease classifier.

Endpoints:
    GET  /health   liveness/readiness probe
    POST /predict  prediction + probability for one patient record
    GET  /metrics  Prometheus metrics (via prometheus-fastapi-instrumentator)
    GET  /docs     interactive Swagger UI

Run locally (from the project root):
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schemas import HealthResponse, PatientData, PredictionResponse
from src.config import ALL_FEATURES, MODEL_METADATA_PATH, MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart_api")

PREDICTIONS = Counter(
    "model_predictions_total",
    "Predictions served, labelled by predicted class",
    ["outcome"],
)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", MODEL_PATH))
    metadata_path = Path(os.getenv("MODEL_METADATA_PATH", MODEL_METADATA_PATH))
    if not model_path.exists():
        raise RuntimeError(
            f"Model not found at {model_path}. Train it first: python -m src.train"
        )
    _state["model"] = joblib.load(model_path)
    _state["metadata"] = (
        json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    )
    logger.info(
        "Loaded model %s from %s",
        _state["metadata"].get("model_name", "unknown"),
        model_path,
    )
    yield
    _state.clear()


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts the risk of heart disease from patient health data "
    "(UCI Heart Disease dataset).",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        '%s %s -> %d (%.1f ms) client=%s',
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request.client.host if request.client else "unknown",
    )
    return response


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": app.title,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded="model" in _state,
        model_name=_state.get("metadata", {}).get("model_name"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData) -> PredictionResponse:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # None (omitted ca/thal) becomes NaN, handled by the pipeline's imputers.
    features = pd.DataFrame([patient.model_dump()])[ALL_FEATURES].astype("float64")
    try:
        probability = float(_state["model"].predict_proba(features)[0, 1])
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from None

    prediction = int(probability >= 0.5)
    PREDICTIONS.labels(outcome=str(prediction)).inc()
    return PredictionResponse(
        prediction=prediction,
        label="Heart disease" if prediction else "No heart disease",
        probability=round(probability, 4),
        model_name=_state.get("metadata", {}).get("model_name", "unknown"),
        trained_at=_state.get("metadata", {}).get("trained_at", "unknown"),
    )
