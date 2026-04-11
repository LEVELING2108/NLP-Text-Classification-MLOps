from fastapi.testclient import TestClient

import api.main as main_api


class DummyPipeline:
    metadata = {"model_version": "test-version"}

    def predict(self, text: str) -> str:
        return "spam" if "win" in text.lower() else "ham"


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
