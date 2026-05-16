import pandas as pd

from mlops_nlp.data.preprocessing import clean_text, preprocess_dataframe


def test_clean_text_removes_noise():
    text = "Visit https://example.com NOW!!!"
    # 'now' is a stop word and is removed
    assert clean_text(text) == "visit"


def test_clean_text_lemmatization():
    text = "the dogs are running"
    # 'the', 'are' are stop words. 'dogs' -> 'dog', 'running' -> 'running' (default lemmatizer needs POS tag for verbs, but dogs -> dog works)
    # Actually WordNetLemmatizer.lemmatize("dogs") -> "dog"
    assert clean_text(text) == "dog running"


def test_preprocess_dataframe_filters_empty_rows():
    frame = pd.DataFrame({"text": ["hello", None, "!!!"], "label": ["ham", "spam", "spam"]})
    output = preprocess_dataframe(frame, text_column="text", target_column="label")
    assert len(output) == 1
    assert output.iloc[0]["text"] == "hello"

