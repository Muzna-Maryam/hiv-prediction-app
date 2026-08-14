# HIV outcome predictor

[![Backend CI](https://github.com/Muzna-Maryam/hiv-prediction-app/actions/workflows/backend-ci.yml/badge.svg)](https://github.com/Muzna-Maryam/hiv-prediction-app/actions/workflows/backend-ci.yml)
[![Frontend CI](https://github.com/Muzna-Maryam/hiv-prediction-app/actions/workflows/frontend-ci.yml/badge.svg)](https://github.com/Muzna-Maryam/hiv-prediction-app/actions/workflows/frontend-ci.yml)

A full-stack, MLOps-backed rebuild of a course project that originally trained
a few scikit-learn models in a single script. This version turns that into a
served, explainable, tracked, tested, containerized, and deployed prediction
system.

**Live demo:** https://hiv-prediction-app-frontend.onrender.com

(backend API: https://hiv-prediction-app-8oo8.onrender.com/docs)

> Hosted on Render's free tier, the backend spins down after ~15 minutes of
> inactivity, so the first request after a while can take 30-60 seconds to
> wake it back up. That's expected, not a bug.

## What it does

Given a set of clinical features from the ACTG175 AIDS Clinical Trial dataset
(demographics, treatment history, CD4/CD8 lab values), predicts the
probability that a patient reaches the study's clinical endpoint, and explains
*why* using SHAP, not just a bare probability, but which specific factors
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
app/main.py The API endpoints + CORS
app/models_registry.py Queries MLflow for the comparison dashboard
tests/ Self-contained test suite, trains its own throwaway
model on synthetic data (tests/conftest.py), so it
needs no real dataset and no manual setup

.github/workflows/ CI: runs the backend test suite and the frontend
type-check + build on every push
Dockerfile (x2 per service) Local dev (volume-mounted, trains on first run)
vs. deploy (model baked in at build time)

## Why it's built this way

- **One Pipeline object, not separate scripts.** Feature engineering,
  scaling, and the model are chained into a single scikit-learn `Pipeline`.
  Training and serving use the exact same object, there's no way for the
  code that prepares data at training time to drift from the code that
  prepares it at prediction time.
- **MLflow tracks every run.** Every retrain logs params, accuracy, and a
  full classification report for all 4 candidate models, and registers
  whichever one wins as the current deployed model. `mlflow ui` shows the
  full history.
- **SHAP is model-agnostic.** Explanations are computed by treating the
  entire pipeline as a black box (`algorithm="permutation"`), not a
  tree-specific method, so explanations keep working even if a future
  retrain picks a different winning model (SVM today, RandomForest
  tomorrow).
- **The `time` feature was deliberately removed.** It's a survival-analysis
  artifact (time to the clinical event or censoring) that's mechanically
  entangled with the label itself, not a genuine predictive signal, see
  the comment in `app/train.py` for the full reasoning. Removing it trades
  some raw accuracy for a number that's actually meaningful.
- **Tests never touch the real dataset.** `tests/conftest.py` trains a
  throwaway model on synthetic data before any test runs, so the suite is
  self-contained - works the same locally, in CI, or on a fresh clone,
  with zero manual setup and no dependency on private data being present.
- **Two different Dockerfiles, on purpose.** Local dev trains the model at
  *container start* and persists it to a mounted volume, the right call
  when you control the disk. The deploy image (`Dockerfile.deploy`) instead
  trains once at *build time* and bakes the result directly in, because
  Render's free tier has no persistent disk to mount.
- **Frontend and backend are deployed as two separate services** (a static
  site and a Docker web service) rather than one combined container, with
  CORS explicitly configured between them, simpler and more portable than
  relying on Docker-network-style service discovery on a platform that
  doesn't expose it the same way `docker compose` does locally.

## Running it locally

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

**Run the tests** (no training or setup needed first):

cd backend
pytest tests/ -v

**View experiment history:**

cd backend
mlflow ui

## Running it with Docker

docker compose up --build

Then open `http://localhost:3000`. First startup trains the model
automatically (empty `artifacts/` on a fresh clone); subsequent restarts
reuse the trained model via the mounted volume.

## Deployment

Deployed on Render's free tier: the backend as a Docker web service
(`backend/Dockerfile.deploy`, model baked in at build time), the frontend as
a static site (built with Vite, calling the backend's public URL directly
with CORS enabled).

## CI

Two GitHub Actions workflows, each path-filtered to only run when the
relevant half of the app changes:
- **Backend CI** - installs dependencies, runs the full self-contained test
  suite (`pytest tests/`).
- **Frontend CI** - installs dependencies, runs `npm run build` (type-checks
  via `tsc -b`, then builds).

## Known limitations

- Recall on the minority class (patients who reach the endpoint) is
  meaningfully lower than overall accuracy across all four models -
  disclosed rather than hidden, since it's the honest weak point of the
  current approach. Worth addressing with `class_weight="balanced"` or
  resampling in a future iteration.
- SHAP's permutation explainer is model-agnostic but slow (seconds per
  request) - fine for this demo, would need a faster explainer for
  real-time production traffic.
- Data versioning (DVC) was on the original plan and deliberately scoped
  out once MLflow's registry covered model versioning - a conscious
  trade, not an oversight.

## Stack

Python, scikit-learn, FastAPI, MLflow, SHAP, React, TypeScript, Vite,
Docker, GitHub Actions, Render.
