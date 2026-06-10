from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlops_nlp.config import load_config
from mlops_nlp.utils.drift import run_drift_check, save_drift_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data drift detection.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config yaml")
    parser.add_argument("--output", type=str, default="data/logs/drift_report.json", help="Path to save report")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_drift_check(config)
    
    if result["status"] == "success":
        save_drift_report(result, args.output)
        print(f"Drift check complete. Results saved to {args.output}")
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
