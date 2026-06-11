from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: str
    confidence: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predictions: list[PredictionResponse]

