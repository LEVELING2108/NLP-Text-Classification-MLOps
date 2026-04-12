from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from sklearn.model_selection import train_test_split

from mlops_nlp.config import load_config
from mlops_nlp.data.ingestion import load_dataset
from mlops_nlp.data.preprocessing import preprocess_dataframe
from mlops_nlp.features.vectorizer import build_vectorizer
from mlops_nlp.logging_config import configure_logging, get_logger
from mlops_nlp.models.evaluate import evaluate_model
from mlops_nlp.models.train import build_classifier
from mlops_nlp.tracking.mlflow_utils import init_mlflow, log_run
from mlops_nlp.utils.io import make_model_version, save_artifacts

LOGGER = get_logger(__name__)


def run_train_pipeline(config_path: str | Path = "configs/config.yaml") -> Dict[str, object]:
    config = load_config(config_path)
    configure_logging(config.monitoring.log_level)
    LOGGER.info("Loading dataset from %s", config.data.raw_path)

    frame = load_dataset(config.data.raw_path)
    frame = preprocess_dataframe(
        frame=frame,
        text_column=config.data.text_column,
        target_column=config.data.target_column,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        frame[config.data.text_column],
        frame[config.data.target_column],
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=frame[config.data.target_column],
    )

    vectorizer = build_vectorizer(config.model)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = build_classifier(config.model)
    model.fit(x_train_vec, y_train)
    predictions = model.predict(x_test_vec)
    metrics = evaluate_model(y_test, predictions)

    model_version = make_model_version()
    metadata = {
        "project_name": config.project.name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "metrics": metrics,
    }

    init_mlflow(config)
    run_id = log_run(
        config=config,
        params={
            "model_type": config.model.type,
            "C": config.model.C,
            "max_iter": config.model.max_iter,
            "ngram_min": config.model.ngram_min,
            "ngram_max": config.model.ngram_max,
            "test_size": config.data.test_size,
        },
        metrics=metrics,
        model=model,
        vectorizer=vectorizer,
        metadata=metadata,
    )

    metadata["mlflow_run_id"] = run_id
    model_path, vectorizer_path, metadata_path = save_artifacts(config, model, vectorizer, metadata)

    LOGGER.info("Training complete with metrics: %s", metrics)
    return {
        "metrics": metrics,
        "model_path": model_path,
        "vectorizer_path": vectorizer_path,
        "metadata_path": metadata_path,
        "run_id": run_id,
    }
