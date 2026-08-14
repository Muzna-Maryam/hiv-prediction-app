"""
API-level smoke test: proves /predict returns a well-formed response for
a valid input, and a 422 for a request missing a required feature. The
model it tests against is trained automatically by conftest.py before
any test runs - no manual setup needed.

Run: pytest tests/
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.fake_data import make_fake_data


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_predict_returns_probability_and_top_features(client):
    X, _ = make_fake_data(n=1, seed=99)
    row = X.iloc[0].to_dict()
    del row["time"]  # dropped as a leaky feature; API no longer accepts it

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
    del row["preanti"]

    response = client.post("/predict", json=row)

    assert response.status_code == 422