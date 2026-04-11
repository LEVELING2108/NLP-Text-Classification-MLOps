from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlops_nlp.pipelines.train_pipeline import run_train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NLP classifier pipeline.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config yaml")
    args = parser.parse_args()

    result = run_train_pipeline(config_path=args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

