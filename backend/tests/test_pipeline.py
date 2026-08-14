"""
Smoke test using synthetic data shaped like AIDS_Classification.csv
(this is the standard ACTG175 clinical trial column set). This doesn't
validate model quality - it only proves the Pipeline builds, fits, and
predicts without errors, BEFORE you run it against the real dataset.
Cheap to run in CI on every push.

Run: pytest tests/
"""

from sklearn.ensemble import RandomForestClassifier

from app.pipeline import build_pipeline
from tests.fake_data import make_fake_data


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