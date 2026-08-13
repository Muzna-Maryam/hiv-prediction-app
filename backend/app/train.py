"""
Training entry point.

Loads the raw CSV, builds one Pipeline per candidate model (see
pipeline.py), tunes each with GridSearchCV, evaluates on a held-out test
set, and saves the best-performing PIPELINE (not just the bare model) to
disk with joblib - so the saved artifact already knows how to turn raw
input rows into a prediction.

CHANGE FROM THE ORIGINAL SCRIPT: added `stratify=y` to the train/test
split. The pie chart in the original EDA suggests the two classes
(infected / not infected) aren't balanced 50/50; without stratifying, a
random split can end up with a test set that under- or over-represents
the minority class, which quietly skews the accuracy number. Stratifying
keeps the class ratio the same in train and test.

Run: python -m app.train
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
print(list(pd.read_csv("data/AIDS_CLASSIFICATION.csv").columns))
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

from app.config import settings
from app.pipeline import build_pipeline

MODELS_PARAM_GRIDS = {
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
        },
    ),
    "LDA": (
        LinearDiscriminantAnalysis(),
        {"clf__solver": ["lsqr", "eigen"], "clf__shrinkage": ["auto", 0.1, 0.5]},
    ),
    "GaussianNB": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]},
    ),
    "SVM": (
        CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
        {
            "clf__estimator__C": [0.1, 1, 10],
            "clf__estimator__kernel": ["linear", "rbf"],
        }
    ),
}


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(settings.data_path)
    # 'time' dropped too: it's mechanically tied to how 'infected' was
    # derived (short time correlates with the event having happened,
    # long time with censoring) - a real deployment scenario would never
    # have "time until the outcome" available as an input anyway.
    return df.drop(columns=["race", "time"])


def main():

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    df = load_raw_data()
    X = df.drop(columns=["infected"])
    y = df["infected"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=40, stratify=y
    )

    best_name, best_score, best_pipeline, best_run_id = None, -1.0, None, None

    for name, (estimator, param_grid) in MODELS_PARAM_GRIDS.items():
        print(f"Tuning {name}...")
        pipeline = build_pipeline(estimator)
        search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
        with mlflow.start_run(run_name=name):
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            mlflow.log_params(search.best_params_)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("recall_infected", report["1"]["recall"])
            mlflow.log_dict(report, "classification_report.json")
            mlflow.sklearn.log_model(
                search.best_estimator_, name="model", serialization_format="cloudpickle"
            )

            print(f"{name}: accuracy={acc:.3f}  best_params={search.best_params_}")
            print(classification_report(y_test, y_pred))

            if acc > best_score:
                best_name = name
                best_score = acc
                best_pipeline = search.best_estimator_
                best_run_id = mlflow.active_run().info.run_id


    print(f"\nBest model: {best_name} ({best_score:.3f})")
    os.makedirs(os.path.dirname(settings.model_path), exist_ok=True)
    joblib.dump(best_pipeline, settings.model_path)

    background_sample = X_train.sample(n=min(100, len(X_train)), random_state=42)
    joblib.dump(background_sample, settings.background_path)
    print(f"Saved SHAP background sample to {settings.background_path}")

    print(f"Saved full pipeline to {settings.model_path}")

    registered = mlflow.register_model(
        f"runs:/{best_run_id}/model", settings.mlflow_model_name
    )
    print(f"Registered as {settings.mlflow_model_name} v{registered.version}")


if __name__ == "__main__":
    main()
