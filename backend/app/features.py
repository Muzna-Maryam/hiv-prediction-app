"""
Feature engineering as a scikit-learn-compatible Transformer.

WHY THIS IS A CLASS, NOT A SCRIPT:
The original script computed log_preanti, healthscore, and squared terms
with plain pandas code that ran once on the whole dataset, before the
train/test split. That's fine for a one-off notebook, but it creates a
"train/serve skew" risk: the FastAPI service that will later serve live
predictions would need to reimplement this exact logic. Any drift between
the two copies silently produces wrong predictions with no error.

Wrapping it as a Transformer lets it be chained into a Pipeline (see
pipeline.py) alongside the scaler and the model. That has three concrete
benefits:
  1. `joblib.dump(pipeline)` serializes feature engineering + scaling +
     model as ONE file. The API loads that file and calls .predict() -
     it never needs to know these engineered features exist.
  2. GridSearchCV/cross-validation apply the transform correctly per fold
     (only ever fit on the training fold), so there's no data leakage.
  3. Changing a feature later means changing it in exactly one place.

ONE DELIBERATE CHANGE FROM THE ORIGINAL:
The original script DROPPED rows where preanti <= 0 before taking log().
That's a defensible choice for offline analysis, but it can't survive
contact with an API: if a single incoming prediction request happens to
have preanti == 0, dropping the row means there's nothing left to predict
on, and the endpoint has to fail instead of answering. So instead of
dropping, this clips preanti to a small positive floor before taking the
log. It changes the numeric value of log_preanti very slightly for
edge-case rows, but it means the transformer always returns exactly as
many rows as it received - required for a single-row API request.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class AidsFeatureEngineer(BaseEstimator, TransformerMixin):
    HEALTH_COLUMNS = ["karnof", "cd40", "cd420", "cd80", "cd820"]
    SQUARE_COLUMNS = ["age", "wtkg", "cd40", "cd420", "cd80", "cd820"]
    PREANTI_FLOOR = 1e-3  # avoids log(0) / log(negative) without dropping rows

    def fit(self, X: pd.DataFrame, y=None):
        # No state to learn here (no mean/std to remember), but fit() must
        # exist and return self for this to be a valid sklearn transformer
        # that Pipeline/GridSearchCV can call.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        safe_preanti = X["preanti"].clip(lower=self.PREANTI_FLOOR)
        X["log_preanti"] = np.log(safe_preanti)

        X["healthscore"] = X[self.HEALTH_COLUMNS].mean(axis=1)

        for col in self.SQUARE_COLUMNS:
            X[f"{col}_squared"] = X[col] ** 2

        return X

    def get_feature_names_out(self, input_features=None):
        # Needed later for SHAP / feature-importance to show real column
        # names instead of "feature_17". sklearn calls this automatically
        # when a Pipeline step downstream asks for output feature names.
        base = list(input_features) if input_features is not None else []
        engineered = ["log_preanti", "healthscore"] + [
            f"{c}_squared" for c in self.SQUARE_COLUMNS
        ]
        return np.array(base + engineered)
