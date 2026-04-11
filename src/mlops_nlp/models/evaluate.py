from __future__ import annotations

from typing import Dict

from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

