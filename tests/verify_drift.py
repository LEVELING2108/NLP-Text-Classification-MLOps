from __future__ import annotations

import json
import os
from pathlib import Path
import pandas as pd
import subprocess

def simulate_inference_logs(log_path: str):
    # Simulate some biased inference logs (mostly spam) to trigger drift
    # Training data usually has more 'ham' than 'spam'
    biased_logs = [
        {"timestamp": "2026-06-09T12:00:00Z", "text": "free money", "prediction": "spam", "confidence": 0.9, "model_version": "test"},
        {"timestamp": "2026-06-09T12:01:00Z", "text": "win prize", "prediction": "spam", "confidence": 0.85, "model_version": "test"},
        {"timestamp": "2026-06-09T12:02:00Z", "text": "click here", "prediction": "spam", "confidence": 0.95, "model_version": "test"},
        {"timestamp": "2026-06-09T12:03:00Z", "text": "hello friend", "prediction": "ham", "confidence": 0.99, "model_version": "test"},
        {"timestamp": "2026-06-09T12:04:00Z", "text": "limited offer", "prediction": "spam", "confidence": 0.7, "model_version": "test"},
    ]
    
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        for entry in biased_logs:
            f.write(json.dumps(entry) + "\n")

def run_test():
    log_path = "data/logs/test_inference.jsonl"
    report_path = "data/logs/test_drift_report.json"
    
    print("Simulating biased inference logs...")
    simulate_inference_logs(log_path)
    
    # Run the drift check pipeline using the simulated logs
    # We'll need to override the config log path or use a temp config
    # For simplicity, we'll just run it and see if it picks up the default log if we move ours there
    
    print("Running drift detection pipeline...")
    # Using environment variable to override if the script supports it, 
    # but our config.py doesn't support MLOPS_INFERENCE_LOG_PATH yet.
    # Let's just temporarily overwrite the default log path in a temp config or use the default one.
    
    # Backup existing log
    original_log = "data/logs/inference.jsonl"
    backup_log = "data/logs/inference.jsonl.bak"
    if os.path.exists(original_log):
        os.rename(original_log, backup_log)
    
    try:
        os.rename(log_path, original_log)
        
        # Run pipeline
        result = subprocess.run(
            [".venv/Scripts/python", "pipelines/run_drift_check.py", "--output", report_path],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"}
        )
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
                print(f"Drift Detected: {report['drift_detected']}")
                print(f"Drift Score: {report['drift_score']}")
        else:
            print("Failed to generate report.")
            
    finally:
        # Restore logs
        if os.path.exists(original_log):
            os.remove(original_log)
        if os.path.exists(backup_log):
            os.rename(backup_log, original_log)

if __name__ == "__main__":
    run_test()
