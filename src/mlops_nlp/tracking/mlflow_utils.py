from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from mlops_nlp.config import AppConfig


def init_mlflow(config: AppConfig) -> None:
    mlflow.set_tracking_uri(config.tracking.mlflow_uri)
    mlflow.set_experiment(config.tracking.experiment_name)


def log_run(
    config: AppConfig,
    params: Dict[str, object],
    metrics: Dict[str, float],
    model,
    vectorizer,
    metadata: Dict[str, object],
    model_input_example,
    model_output_example,
) -> str:
    with mlflow.start_run(run_name=config.tracking.run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        signature = infer_signature(model_input_example, model_output_example)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=model_input_example,
            registered_model_name=config.tracking.registered_model_name,
        )
        metadata_with_run = dict(metadata)
        metadata_with_run["mlflow_run_id"] = run.info.run_id
        mlflow.log_dict(metadata_with_run, "inference_bundle/metadata.json")
        with TemporaryDirectory() as temp_dir:
            vectorizer_path = Path(temp_dir) / "vectorizer.joblib"
            joblib.dump(vectorizer, vectorizer_path)
            mlflow.log_artifact(str(vectorizer_path), artifact_path="inference_bundle")
        return run.info.run_id
