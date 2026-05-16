from __future__ import annotations

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import pandas as pd

# Pre-compile patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA_NUM_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

# Initialize NLTK resources
_RESOURCES_DOWNLOADED = False

def _ensure_nltk_resources():
    global _RESOURCES_DOWNLOADED
    if not _RESOURCES_DOWNLOADED:
        for res in ["punkt", "wordnet", "stopwords", "punkt_tab"]:
            nltk.download(res, quiet=True)
        _RESOURCES_DOWNLOADED = True

_ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    # Basic cleaning
    value = text.strip().lower()
    value = URL_PATTERN.sub("", value)
    value = NON_ALPHA_NUM_PATTERN.sub(" ", value)
    value = MULTI_SPACE_PATTERN.sub(" ", value)
    
    # Tokenization, stop word removal, and lemmatization
    tokens = word_tokenize(value)
    cleaned_tokens = [
        LEMMATIZER.lemmatize(token) 
        for token in tokens 
        if token not in STOP_WORDS
    ]
    
    return " ".join(cleaned_tokens).strip()


def preprocess_dataframe(
    frame: pd.DataFrame, text_column: str, target_column: str
) -> pd.DataFrame:
    data = frame[[text_column, target_column]].dropna().copy()
    data[text_column] = data[text_column].astype(str).map(clean_text)
    return data[data[text_column].str.len() > 0]

