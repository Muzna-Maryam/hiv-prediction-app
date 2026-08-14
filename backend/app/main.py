"""
FastAPI app. Run: uvicorn app.main:app --reload
"""
import os
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from app.explain import explain_prediction, explain_prediction_full, get_explainer
from app.models_registry import get_latest_model_comparison
from app.schemas import ExplanationResponse, ModelRun, PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the pipeline + build the SHAP explainer once at startup, not on
    # the first request - so the first real user doesn't pay the (multi-
    # second) explainer setup cost.
    get_explainer()
    yield


app = FastAPI(title="HIV Outcome Predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("ALLOWED_ORIGIN", "*")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    # No more manual "missing feature" handling needed here - pydantic
    # itself returns a 422 with the exact missing/invalid field before
    # this function body even runs.
    input_df = pd.DataFrame([payload.model_dump()])
    return explain_prediction(input_df)

@app.post("/explain", response_model=ExplanationResponse)
def explain(payload: PredictionRequest):
    input_df = pd.DataFrame([payload.model_dump()])
    return explain_prediction_full(input_df)


@app.get("/models/comparison", response_model=list[ModelRun])
def models_comparison():
    return get_latest_model_comparison()