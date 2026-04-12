from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from mlops_nlp.config import ModelConfig


def build_classifier(model_config: ModelConfig) -> LogisticRegression:
    if model_config.type != "logistic_regression":
        raise ValueError(
            f"Unsupported model.type '{model_config.type}'. Only 'logistic_regression' is supported."
        )
    return LogisticRegression(C=model_config.C, max_iter=model_config.max_iter)
