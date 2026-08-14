"""
Central config, replacing the hardcoded

    file_path = "data/AIDS_Classification.csv"

from the original script. Pydantic's BaseSettings reads from environment
variables or a .env file, with the defaults below as a fallback. This means
the exact same code works on your Mac, in Docker, and in CI - only the
environment differs, never the code.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    data_path: str = "data/AIDS_Classification.csv"
    model_path: str = "artifacts/best_pipeline.joblib"
    background_path: str = "artifacts/background_sample.joblib"

    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_experiment_name: str = "hiv-outcome-prediction"
    mlflow_model_name: str = "hiv-outcome-predictor"


settings = Settings()
