from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from mlops_nlp.config import ModelConfig


def build_vectorizer(model_config: ModelConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(model_config.ngram_min, model_config.ngram_max),
        min_df=model_config.min_df,
    )

