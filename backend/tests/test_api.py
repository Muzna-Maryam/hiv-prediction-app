"""
API-level smoke test: proves /predict returns a well-formed response for
a valid input, and a 400 (not a 500) for a request missing a required
feature. Needs artifacts/best_pipeline.joblib and background_sample.joblib
to already exist - run `python -m app.train` first.

Run: pytest tests/
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.test_pipeline import make_fake_data


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_predict_returns_probability_and_top_features(client):
    X, _ = make_fake_data(n=1, seed=99)
    row = X.iloc[0].to_dict()

    response = client.post("/predict", json=row)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability_infected"] <= 1.0
    assert body["predicted_label"] in ("infected", "not_infected")
    assert len(body["top_features"]) == 5
    assert {"feature", "contribution"} <= body["top_features"][0].keys()


def test_predict_missing_feature_returns_422(client):
    X, _ = make_fake_data(n=1, seed=99)
    row = X.iloc[0].to_dict()
    del row["time"]
    del row["preanti"]  # required by AidsFeatureEngineer

    response = client.post("/predict", json=row)

    assert response.status_code == 422