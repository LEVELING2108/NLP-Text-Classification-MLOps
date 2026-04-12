from fastapi.testclient import TestClient
import pytest

import api.main as main_api


class DummyPipeline:
    metadata = {"model_version": "test-version"}

    def predict(self, text: str) -> str:
        return "spam" if "win" in text.lower() else "ham"


@pytest.fixture(autouse=True)
def reset_pipeline(monkeypatch):
    monkeypatch.setattr(main_api, "pipeline", None)


def test_health_endpoint():
    client = TestClient(main_api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_success(monkeypatch):
    monkeypatch.setattr(main_api, "pipeline", DummyPipeline())
    client = TestClient(main_api.app)
    response = client.post("/predict", json={"text": "You WIN big prize"})
    assert response.status_code == 200
    assert response.json()["prediction"] == "spam"
    assert response.json()["model_version"] == "test-version"


def test_predict_validation_error():
    client = TestClient(main_api.app)
    response = client.post("/predict", json={"text": "x"})
    assert response.status_code == 422


def test_predict_model_not_loaded():
    client = TestClient(main_api.app)
    response = client.post("/predict", json={"text": "valid request text"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded. Train model first."


def test_metrics_endpoint_available():
    client = TestClient(main_api.app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
