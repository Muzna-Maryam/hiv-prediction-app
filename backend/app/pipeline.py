"""
Builds the single Pipeline object used for BOTH training and serving.

WHY ONE PIPELINE OBJECT INSTEAD OF SEPARATE STEPS IN train.py:
Every step here (engineer features -> scale -> select k best -> classify)
has to run identically at training time and at prediction time - fit on
the training data once, then only ever .transform() after that, never
re-fit. A scikit-learn Pipeline enforces this automatically: GridSearchCV
tunes the whole thing as one unit, and the exact pipeline you save is the
exact pipeline the API loads later. There's no manual "remember to scale
the input the same way before predicting" step to forget in the API code.

NOTE ON GridSearchCV PARAM NAMES:
Because the classifier is now a step named "clf" inside the Pipeline,
grid search param keys need a "clf__" prefix, e.g. what was
'n_estimators' becomes 'clf__n_estimators'. See train.py.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from app.features import AidsFeatureEngineer


def build_pipeline(estimator, k_features="all") -> Pipeline:
    return Pipeline(
        [
            ("features", AidsFeatureEngineer()),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k_features)),
            ("clf", estimator),
        ]
    )
