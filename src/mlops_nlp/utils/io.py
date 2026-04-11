from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib

from mlops_nlp.config import AppConfig


def make_model_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def save_artifacts(config: AppConfig, model, vectorizer, metadata: Dict[str, Any]) -> Tuple[str, str, str]:
    model_dir = Path(config.artifacts.model_dir)
    version = metadata["model_version"]
    version_dir = model_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / config.artifacts.model_name
    vectorizer_path = version_dir / config.artifacts.vectorizer_name
    metadata_path = version_dir / config.artifacts.metadata_name

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    latest = model_dir / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "vectorizer_path": str(vectorizer_path),
                "metadata_path": str(metadata_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(model_path), str(vectorizer_path), str(metadata_path)


def load_latest_paths(model_dir: str) -> Dict[str, str]:
    latest_file = Path(model_dir) / "latest.json"
    if not latest_file.exists():
        raise FileNotFoundError("No trained model found. Run training first.")
    return json.loads(latest_file.read_text(encoding="utf-8"))


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))

