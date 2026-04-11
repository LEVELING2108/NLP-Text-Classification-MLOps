from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000)


class PredictionResponse(BaseModel):
    prediction: str
    model_version: str

