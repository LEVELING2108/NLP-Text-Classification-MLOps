from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mlops_nlp.logging_config import get_logger

LOGGER = get_logger(__name__)

def log_inference(
    log_path: str | Path,
    text: str,
    prediction: str,
    confidence: float,
    model_version: str,
) -> None:
    """Logs inference data to a JSONL file for future drift detection."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "prediction": prediction,
        "confidence": confidence,
        "model_version": model_version,
    }
    
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        LOGGER.error("Failed to log inference data: %s", e)
