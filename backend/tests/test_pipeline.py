"""
Smoke test using synthetic data shaped like AIDS_CLASSIFICATION.csv
(this is the standard ACTG175 clinical trial column set). This doesn't
validate model quality - it only proves the Pipeline builds, fits, and
predicts without errors, BEFORE you run it against the real dataset.
Cheap to run in CI on every push.

Run: pytest tests/
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.pipeline import build_pipeline


def make_fake_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "time": rng.integers(1, 1200, n),
            "trt": rng.integers(0, 4, n),
            "age": rng.integers(18, 70, n),
            "wtkg": rng.uniform(40, 120, n),
            "hemo": rng.integers(0, 2, n),
            "homo": rng.integers(0, 2, n),
            "drugs": rng.integers(0, 2, n),
            "karnof": rng.choice([70, 80, 90, 100], n),
            "oprior": rng.integers(0, 2, n),
            "z30": rng.integers(0, 2, n),
            "zprior": rng.integers(0, 2, n),
            "preanti": rng.integers(0, 2000, n),
            "gender": rng.integers(0, 2, n),
            "str2": rng.integers(0, 2, n),
            "strat": rng.integers(1, 4, n),
            "symptom": rng.integers(0, 2, n),
            "treat": rng.integers(0, 2, n),
            "offtrt": rng.integers(0, 2, n),
            "cd40": rng.uniform(0, 500, n),
            "cd420": rng.uniform(0, 500, n),
            "cd80": rng.uniform(0, 2000, n),
            "cd820": rng.uniform(0, 2000, n),
        }
    )
    y = pd.Series(rng.integers(0, 2, n), name="infected")
    return X, y


def test_pipeline_fits_and_predicts():
    X, y = make_fake_data()
    pipeline = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=0))
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    assert len(preds) == len(y)

    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(y), 2)


def test_pipeline_handles_zero_preanti_without_dropping_rows():
    # This specifically checks the log(0) edge case the feature engineer
    # has to survive for a single-row API request (see features.py docstring).
    X, y = make_fake_data(n=50, seed=1)
    X.loc[0, "preanti"] = 0
    pipeline = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=0))
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(X)  # no row silently dropped
