#!/bin/sh
set -e

# artifacts/ is a mounted volume (see docker-compose.yml), so it
# persists across container restarts and rebuilds. First-ever start
# won't have a model yet - train once, automatically, so `docker compose
# up` on a fresh clone just works without a separate manual step.
# Restarts after that skip straight to serving, since the model already
# exists on the mounted volume.
if [ ! -f "artifacts/best_pipeline.joblib" ]; then
  echo "No trained model found in artifacts/ - training now..."
  python -m app.train
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000