from __future__ import annotations

from dataclasses import dataclass
import json

import joblib

from mlops_nlp.data.preprocessing import clean_text


@dataclass
class ModelBundle:
    model: object
    vectorizer: object
    metadata: dict

    def predict(self, text: str) -> str:
        cleaned = clean_text(text)
        vector = self.vectorizer.transform([cleaned])
        return str(self.model.predict(vector)[0])


def load_bundle(model_path: str, vectorizer_path: str, metadata_path: str) -> ModelBundle:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return ModelBundle(model=model, vectorizer=vectorizer, metadata=metadata)
