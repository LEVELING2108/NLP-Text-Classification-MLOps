from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from mlops_nlp.config import ModelConfig


def build_classifier(model_config: ModelConfig) -> LogisticRegression:
    return LogisticRegression(C=model_config.C, max_iter=model_config.max_iter)

