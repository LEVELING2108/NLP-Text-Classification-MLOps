from pathlib import Path
from mlops_nlp.utils.drift import log_inference

def test_log_inference_creates_file(tmp_path: Path):
    log_file = tmp_path / "inference.jsonl"
    log_inference(
        log_path=log_file,
        text="test message",
        prediction="ham",
        confidence=0.95,
        model_version="v1"
    )
    
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "test message" in content
    assert '"prediction": "ham"' in content
    assert '"confidence": 0.95' in content
    assert '"model_version": "v1"' in content
