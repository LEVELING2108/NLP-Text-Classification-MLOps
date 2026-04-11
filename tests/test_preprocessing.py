import pandas as pd

from mlops_nlp.data.preprocessing import clean_text, preprocess_dataframe


def test_clean_text_removes_noise():
    text = "Visit https://example.com NOW!!!"
    assert clean_text(text) == "visit now"


def test_preprocess_dataframe_filters_empty_rows():
    frame = pd.DataFrame({"text": ["hello", None, "!!!"], "label": ["ham", "spam", "spam"]})
    output = preprocess_dataframe(frame, text_column="text", target_column="label")
    assert len(output) == 1
    assert output.iloc[0]["text"] == "hello"

