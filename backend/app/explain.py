"""
SHAP explanation for the served model.

WHY A MODEL-AGNOSTIC EXPLAINER, NOT shap.TreeExplainer:
TreeExplainer is fast and exact, but only works on tree-based models
(RandomForest, XGBoost). Which model wins is decided purely by accuracy at
train time (train.py) - today it's SVM, last run it was a coin flip
between SVM and RandomForest. A tree-specific explainer would silently
need swapping the moment a non-tree model wins a future retrain.

shap.Explainer(pipeline.predict_proba, background, algorithm="permutation")
treats the ENTIRE pipeline - feature engineering, scaling, selection,
classifier - as one black-box function: raw input row in, probability out.
That means it works identically no matter which model is currently
deployed, and the returned attributions are in terms of your original
raw input columns (age, cd40, ...), not the scaled/selected internal
features - which is what you want to show on a form the user filled in
with real values, not standardized ones.

TRADE-OFF: permutation explanation re-runs the whole pipeline many times
per prediction (once per feature, roughly), so it's noticeably slower
than a single .predict() call - seconds, not milliseconds. Fine for a
demo/portfolio API; would need swapping to a faster, model-specific
explainer if this ever needed to serve real-time traffic.
"""

import joblib
import pandas as pd
import shap

from app.config import settings

_pipeline = None
_background = None
_explainer = None


def get_explainer():
    """Loads the pipeline + background sample once and caches the SHAP
    explainer, so the (slow-ish) setup cost is paid at startup, not on
    every request."""
    global _pipeline, _background, _explainer
    if _explainer is None:
        _pipeline = joblib.load(settings.model_path)
        _background = joblib.load(settings.background_path)
        _explainer = shap.Explainer(
            _pipeline.predict_proba, _background, algorithm="permutation"
        )
    return _explainer, _pipeline


def explain_prediction(input_df: pd.DataFrame, top_n: int = 5) -> dict:
    explainer, pipeline = get_explainer()

    proba = pipeline.predict_proba(input_df)[0]
    predicted_class = int(proba.argmax())

    shap_values = explainer(input_df)
    # .values shape is (n_rows, n_features, n_outputs) because predict_proba
    # returns 2 columns; index 1 = attribution toward the "infected" class.
    contributions = shap_values.values[0, :, 1]

    ranked = sorted(
        zip(input_df.columns.tolist(), contributions),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )[:top_n]

    return {
        "probability_infected": float(proba[1]),
        "predicted_label": "infected" if predicted_class == 1 else "not_infected",
        "top_features": [
            {"feature": name, "contribution": float(value)} for name, value in ranked
        ],
    }

def explain_prediction_full(input_df: pd.DataFrame) -> dict:
    """Every feature's contribution (not just top 5) plus base_value -
    needed for the Explain page's waterfall: base_value -> each feature
    push -> final probability."""
    explainer, pipeline = get_explainer()
    proba = pipeline.predict_proba(input_df)[0]
    predicted_class = int(proba.argmax())

    shap_values = explainer(input_df)
    contributions = shap_values.values[0, :, 1]
    base_value = float(shap_values.base_values[0, 1])

    ranked = sorted(
        zip(input_df.columns.tolist(), contributions),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )

    return {
        "probability_infected": float(proba[1]),
        "predicted_label": "infected" if predicted_class == 1 else "not_infected",
        "base_value": base_value,
        "all_features": [{"feature": n, "contribution": float(v)} for n, v in ranked],
    }