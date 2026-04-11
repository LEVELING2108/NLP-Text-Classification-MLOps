from pathlib import Path

import yaml

from mlops_nlp.pipelines.train_pipeline import run_train_pipeline


def test_train_pipeline_end_to_end(tmp_path: Path):
    data_file = tmp_path / "dataset.csv"
    data_file.write_text(
        "\n".join(
            [
                "text,label",
                "free cash reward now,spam",
                "team meeting at noon,ham",
                "claim your prize,spam",
                "please review the report,ham",
                "lottery winner click here,spam",
                "project update shared,ham",
            ]
        ),
        encoding="utf-8",
    )

    config = {
        "project": {"name": "test-project", "task": "spam-detection"},
        "data": {
            "raw_path": str(data_file),
            "text_column": "text",
            "target_column": "label",
            "test_size": 0.33,
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
            "model_dir": str(tmp_path / "models"),
            "model_name": "classifier.joblib",
            "vectorizer_name": "vectorizer.joblib",
            "metadata_name": "metadata.json",
        },
        "tracking": {
            "mlflow_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "experiment_name": "unit-test-exp",
            "run_name": "unit-test-run",
        },
        "api": {"title": "x", "version": "1", "description": "x"},
        "monitoring": {"enable_prometheus": True, "log_level": "INFO"},
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_train_pipeline(config_file)
    assert "metrics" in result
    assert result["metrics"]["accuracy"] >= 0.0
    assert Path(result["model_path"]).exists()
    assert Path(result["vectorizer_path"]).exists()
    assert Path(result["metadata_path"]).exists()

