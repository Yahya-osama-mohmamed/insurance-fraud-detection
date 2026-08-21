"""Fraud-scoring API.

POST /predict  -> {fraud_probability, flagged, threshold, top_reasons}
GET  /health   -> model status
GET  /queue    -> the 20 most recent flagged claims (in-memory review queue)
"""

import datetime
from collections import deque
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.model_loader import MODEL_PATH, load
from api.schemas import ClaimInput, HealthResponse, PredictionResponse, QueueItem
from src.explain import reason_strings, shap_values_for

app = FastAPI(
    title="Insurance Claims Fraud Detection API",
    description="Scores claims for fraud risk and explains every flag with SHAP.",
    version="1.0.0",
)

# Minimal analyst UI: a single static page served by the API itself —
# no Streamlit, no build step, no extra dependency
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")

REVIEW_QUEUE: deque = deque(maxlen=20)


@app.get("/health", response_model=HealthResponse)
def health():
    pipeline, _, _ = load()
    version = ("not loaded — run the training pipeline" if pipeline is None else
               datetime.datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat())
    return HealthResponse(
        status="healthy" if pipeline is not None else "degraded",
        model_version=version,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(claim: ClaimInput):
    pipeline, explainer, threshold = load()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    df = pd.DataFrame([claim.model_dump()])
    prob = float(pipeline.predict_proba(df)[0, 1])
    flagged = prob >= threshold

    sv, X_t = shap_values_for(pipeline, explainer, df)
    reasons = reason_strings(sv[0], X_t.iloc[0], top_n=3)

    if flagged:
        REVIEW_QUEUE.appendleft(QueueItem(
            received_at=datetime.datetime.now(datetime.UTC).isoformat(),
            fraud_probability=round(prob, 4),
            top_reasons=reasons,
        ))

    return PredictionResponse(
        fraud_probability=round(prob, 4),
        flagged=flagged,
        threshold=threshold,
        top_reasons=reasons,
    )


@app.get("/queue", response_model=list[QueueItem])
def queue():
    """The 20 most recent flagged claims — a minimal analyst review queue."""
    return list(REVIEW_QUEUE)
