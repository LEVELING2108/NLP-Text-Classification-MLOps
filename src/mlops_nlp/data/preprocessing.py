from __future__ import annotations

import re

import pandas as pd

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA_NUM_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    value = text.strip().lower()
    value = URL_PATTERN.sub("", value)
    value = NON_ALPHA_NUM_PATTERN.sub(" ", value)
    value = MULTI_SPACE_PATTERN.sub(" ", value)
    return value.strip()


def preprocess_dataframe(
    frame: pd.DataFrame, text_column: str, target_column: str
) -> pd.DataFrame:
    data = frame[[text_column, target_column]].dropna().copy()
    data[text_column] = data[text_column].astype(str).map(clean_text)
    return data[data[text_column].str.len() > 0]

