from pathlib import Path

import yaml

from mlops_nlp.config import load_config


def test_load_config_applies_env_overrides(monkeypatch, tmp_path: Path):
    config = {
        "project": {"name": "test-project", "task": "spam-detection"},
        "data": {
            "raw_path": "data/raw/spam_dataset.csv",
            "text_column": "text",
            "target_column": "label",
            "test_size": 0.2,
            "random_state": 42,
        },
        "model": {
            "type": "logistic_regression",
            "C": 1.0,
            "max_iter": 200,
            "ngram_min": 1,
            "ngram_max": 2,
            "min_df": 1,
        },
        "artifacts": {
            "model_dir": "models",
            "model_name": "classifier.joblib",
            "vectorizer_name": "vectorizer.joblib",
            "metadata_name": "metadata.json",
        },
        "tracking": {
            "mlflow_uri": "sqlite:///mlflow.db",
            "experiment_name": "unit-test-exp",
            "run_name": "unit-test-run",
        },
        "api": {"title": "x", "version": "1", "description": "x"},
        "monitoring": {"enable_prometheus": True, "log_level": "INFO"},
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setenv("MLOPS_MLFLOW_URI", "sqlite:///override.db")
    monkeypatch.setenv("MLOPS_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MLOPS_ENABLE_PROMETHEUS", "false")

    loaded = load_config(config_file)

    assert loaded.tracking.mlflow_uri == "sqlite:///override.db"
    assert loaded.monitoring.log_level == "DEBUG"
    assert loaded.monitoring.enable_prometheus is False
