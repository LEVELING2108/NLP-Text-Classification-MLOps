from __future__ import annotations

import json

import joblib

from mlops_nlp.config import load_config
from mlops_nlp.data.preprocessing import clean_text
from mlops_nlp.utils.io import load_latest_paths


class InferencePipeline:
    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config = load_config(config_path)
        latest = load_latest_paths(self.config.artifacts.model_dir)
        self.model = joblib.load(latest["model_path"])
        self.vectorizer = joblib.load(latest["vectorizer_path"])
        with open(latest["metadata_path"], "r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

    def predict(self, text: str) -> str:
        cleaned = clean_text(text)
        features = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(features)[0]
        return str(prediction)

