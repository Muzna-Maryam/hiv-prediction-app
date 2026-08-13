# HIV outcome predictor

A full-stack, MLOps-backed rebuild of a course project that originally trained
a few scikit-learn models in a single script. This version turns that into a
served, explainable, tracked prediction system.

## What it does

Given a set of clinical features from the ACTG175 AIDS Clinical Trial dataset
(demographics, treatment history, CD4/CD8 lab values), predicts the
probability that a patient reaches the study's clinical endpoint, and explains
*why* using SHAP - not just a bare probability, but which specific factors
pushed the prediction up or down.

## Architecture

frontend/ React + TypeScript (Vite)
Predict page -> POST /predict -> probability + top 5 contributing factors
Explain page -> POST /explain -> full 20-feature SHAP breakdown + base value
Compare page -> GET /models/comparison -> live MLflow run comparison

backend/ FastAPI + scikit-learn + SHAP + MLflow
app/features.py Custom sklearn Transformer for feature engineering
app/pipeline.py Builds the full Pipeline (features -> scale -> select -> model)
app/train.py Trains 4 candidate models, tracks each in MLflow, saves the best
app/explain.py Model-agnostic SHAP explainer wrapping the whole pipeline
app/main.py The 3 API endpoints
app/models_registry.py Queries MLflow for the comparison dashboard

## Why it's built this way

- **One Pipeline object, not separate scripts.** Feature engineering,
  scaling, and the model are chained into a single scikit-learn `Pipeline`.
  Training and serving use the exact same object - there's no way for the
  code that prepares data at training time to drift from the code that
  prepares it at prediction time.
- **MLflow tracks every run.** Every retrain logs params, accuracy, and a
  full classification report for all 4 candidate models, and registers
  whichever one wins as the current deployed model. `mlflow ui` shows the
  full history.
- **SHAP is model-agnostic.** Explanations are computed by treating the
  entire pipeline as a black box (`algorithm="permutation"`), not a
  tree-specific method - so explanations keep working even if a future
  retrain picks a different winning model (SVM today, RandomForest
  tomorrow).
- **The `time` feature was deliberately removed.** It's a survival-analysis
  artifact (time to the clinical event or censoring) that's mechanically
  entangled with the label itself, not a genuine predictive signal - see
  the comment in `app/train.py` for the full reasoning. Removing it trades
  some raw accuracy for a number that's actually meaningful.

## Running it

**Backend:**

cd backend
pip install -r requirements.txt
python -m app.train # trains, tracks in MLflow, saves the model
uvicorn app.main:app --reload


**Frontend** (in a second terminal):

cd frontend
npm install
npm run dev


Then open `http://localhost:5173`. The dev server proxies API calls to the
backend on `:8000`.

**View experiment history:**

cd backend
mlflow ui


## Known limitations

- Recall on the minority class (patients who reach the endpoint) is
  meaningfully lower than overall accuracy across all four models -
  disclosed rather than hidden, since it's the honest weak point of the
  current approach. Worth addressing with `class_weight="balanced"` or
  resampling in a future iteration.
- SHAP's permutation explainer is model-agnostic but slow (seconds per
  request) - fine for this demo, would need a faster explainer for
  real-time production traffic.

## Stack

Python, scikit-learn, FastAPI, MLflow, SHAP, React, TypeScript, Vite.