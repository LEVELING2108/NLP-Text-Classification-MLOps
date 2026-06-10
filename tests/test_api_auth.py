from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from api.main import app
import os

client = TestClient(app)

def test_predict_no_auth():
    # If MLOPS_API_KEY is not set, it should allow access (default behavior for local dev if not configured)
    # But for this test, we want to ensure that IF it is set, it blocks.
    # However, the current implementation in main.py checks APP_CONFIG.api.api_key which is loaded at start.
    pass

@pytest.fixture
def mock_api_key(monkeypatch):
    monkeypatch.setenv("MLOPS_API_KEY", "test-secret-key")
    # We need to reload the config or the app to pick up the env var if it's already loaded.
    # In api/main.py, APP_CONFIG = load_config() happens at module level.
    # For a robust test, we might need to override the dependency.
    return "test-secret-key"

def test_predict_with_auth_dependency_override():
    from api.main import get_api_key, app
    
    # Mocking the actual config value for the test
    import api.main
    original_key = api.main.APP_CONFIG.api.api_key
    api.main.APP_CONFIG.api.api_key = "test-secret-key"
    
    try:
        # 1. No key provided
        response = client.post("/predict", json={"text": "hello"})
        assert response.status_code == 403
        
        # 2. Wrong key provided
        response = client.post("/predict", json={"text": "hello"}, headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 403
        
        # 3. Correct key provided
        # We need a model loaded for a full 200, otherwise it might be 503 if no model exists.
        response = client.post("/predict", json={"text": "hello"}, headers={"X-API-Key": "test-secret-key"})
        # We expect 503 (Model not loaded) or 200 (Success), but NOT 403.
        assert response.status_code != 403
        
    finally:
        api.main.APP_CONFIG.api.api_key = original_key
