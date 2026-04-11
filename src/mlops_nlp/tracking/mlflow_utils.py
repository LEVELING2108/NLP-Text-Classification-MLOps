from __future__ import annotations

from typing import Dict

import mlflow
import mlflow.sklearn

from mlops_nlp.config import AppConfig


def init_mlflow(config: AppConfig) -> None:
    mlflow.set_tracking_uri(config.tracking.mlflow_uri)
    mlflow.set_experiment(config.tracking.experiment_name)


def log_run(config: AppConfig, params: Dict[str, object], metrics: Dict[str, float], model) -> str:
    with mlflow.start_run(run_name=config.tracking.run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        return run.info.run_id

