"""
Runs once, before pytest collects any test module, and trains a
throwaway model on synthetic data so `pytest` works standalone - locally
or in CI - with no manual setup and no dependency on the real (private,
gitignored) dataset.

WHY pytest_configure() AND NOT A NORMAL FIXTURE:
app.config's `settings` object is built once, at import time, by reading
whatever environment variables exist at that moment. If this ran as a
regular (even session-scoped, autouse) fixture, pytest would already
have imported test_pipeline.py and test_api.py during collection - both
of which import app.pipeline / app.main, which import app.config - so
`settings` would already be locked in using the real default paths
before this fixture ever got a chance to override them via env vars.
pytest_configure() runs before collection starts, which is early enough
to set the env vars first and have them actually take effect.
"""

import os
import tempfile

from tests.fake_data import make_fake_data


def pytest_configure(config):
    tmpdir = tempfile.mkdtemp(prefix="hiv_predictor_test_")

    csv_path = os.path.join(tmpdir, "fake.csv")
    X, y = make_fake_data(n=300, seed=42)
    df = X.copy()
    df["infected"] = y
    df["race"] = 0
    df.to_csv(csv_path, index=False)

    os.environ["DATA_PATH"] = csv_path
    os.environ["MODEL_PATH"] = os.path.join(tmpdir, "pipeline.joblib")
    os.environ["BACKGROUND_PATH"] = os.path.join(tmpdir, "background.joblib")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tmpdir}/mlflow.db"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "ci-test-run"

    # Imported here, not at module level - has to happen AFTER the env
    # vars above are set, since this import is what triggers app.config
    # to build its Settings object.
    from app.train import main as train_main

    train_main()