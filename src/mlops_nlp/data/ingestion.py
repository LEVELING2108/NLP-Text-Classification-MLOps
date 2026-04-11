from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(raw_path: str | Path) -> pd.DataFrame:
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)

