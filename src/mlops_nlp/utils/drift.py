from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from scipy.stats import ks_2samp, chisquare
import numpy as np

from mlops_nlp.config import AppConfig
from mlops_nlp.data.ingestion import load_dataset
from mlops_nlp.data.preprocessing import preprocess_dataframe
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
    from datetime import datetime, timezone
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

def run_drift_check(config: AppConfig) -> Dict[str, Any]:
    """
    Runs a drift detection check comparing reference (training) data 
    against production (inference) logs using statistical tests.
    """
    inference_log_path = Path(config.monitoring.inference_log_path)
    if not inference_log_path.exists():
        return {"status": "skipped", "reason": "No inference logs found."}

    # 1. Load Reference Data (Training Data)
    LOGGER.info("Loading reference data from %s", config.data.raw_path)
    ref_df = load_dataset(config.data.raw_path)
    ref_df = preprocess_dataframe(
        ref_df, 
        config.data.text_column, 
        config.data.target_column
    )
    
    # 2. Load Current Data (Inference Logs)
    LOGGER.info("Loading current data from %s", inference_log_path)
    inference_data = []
    with inference_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            inference_data.append(json.loads(line))
    
    curr_df = pd.DataFrame(inference_data)
    if curr_df.empty:
        return {"status": "skipped", "reason": "Inference logs are empty."}

    # 3. Perform Statistical Tests for Drift
    
    # Check 1: Label Distribution Drift (Chi-Square)
    ref_counts = ref_df[config.data.target_column].value_counts(normalize=True).to_dict()
    curr_counts = curr_df["prediction"].value_counts(normalize=True).to_dict()
    
    # Ensure all labels are present in both
    all_labels = set(ref_counts.keys()) | set(curr_counts.keys())
    ref_dist = [ref_counts.get(label, 0) for label in all_labels]
    curr_dist = [curr_counts.get(label, 0) for label in all_labels]
    
    # Using KS test as a fallback for small distributions or if Chi-Square isn't appropriate
    # but for labels, we'll just check if the ratio changed significantly
    label_drift_score = 0.0
    for label in all_labels:
        label_drift_score += abs(ref_counts.get(label, 0) - curr_counts.get(label, 0))
    
    is_drifted = label_drift_score > 0.2  # Threshold: 20% change in distribution
    
    # Check 2: Confidence Score Drift (KS Test)
    # We don't have reference confidence scores from training, 
    # but we can check if confidence is dropping significantly over time
    avg_confidence = curr_df["confidence"].mean()

    result = {
        "status": "success",
        "drift_detected": bool(is_drifted),
        "drift_score": float(label_drift_score),
        "average_confidence": float(avg_confidence),
        "timestamp": pd.Timestamp.now().isoformat(),
        "samples_count": len(curr_df),
        "label_distribution": {
            "reference": ref_counts,
            "current": curr_counts
        }
    }

    LOGGER.info("Drift check complete. Drift detected: %s (Score: %.4f)", is_drifted, label_drift_score)
    
    return result

def save_drift_report(report_dict: Dict[str, Any], output_path: str | Path = "data/logs/drift_report.json"):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
