"""Queries MLflow for the model-comparison dashboard. Deduped to the
latest run per model name, since every retrain adds 4 more runs to the
experiment - the dashboard should show the current state, not a
growing history."""

from mlflow.tracking import MlflowClient
from app.config import settings


def get_latest_model_comparison() -> list[dict]:
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
    if experiment is None:
        return []

    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"])

    seen, latest = set(), []
    for run in runs:
        name = run.data.tags.get("mlflow.runName", "unknown")
        if name in seen:
            continue
        seen.add(name)
        latest.append({
            "model_name": name,
            "accuracy": run.data.metrics.get("accuracy", 0.0),
            "recall_infected": run.data.metrics.get("recall_infected", 0.0),
            "params": run.data.params,
        })
    return sorted(latest, key=lambda r: r["accuracy"], reverse=True)