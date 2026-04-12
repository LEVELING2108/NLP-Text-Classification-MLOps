from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str
    task: str


class DataConfig(BaseModel):
    raw_path: str
    text_column: str
    target_column: str
    test_size: float = Field(ge=0.05, le=0.5)
    random_state: int = 42


class ModelConfig(BaseModel):
    type: str
    C: float = 1.0
    max_iter: int = 300
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 1


class ArtifactConfig(BaseModel):
    model_dir: str
    model_name: str
    vectorizer_name: str
    metadata_name: str


class TrackingConfig(BaseModel):
    mlflow_uri: str
    experiment_name: str
    run_name: str


class ApiConfig(BaseModel):
    title: str
    version: str
    description: str


class MonitoringConfig(BaseModel):
    enable_prometheus: bool = True
    log_level: str = "INFO"


class AppConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    artifacts: ArtifactConfig
    tracking: TrackingConfig
    api: ApiConfig
    monitoring: MonitoringConfig


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config(config_path: str | Path = "configs/config.yaml") -> AppConfig:
    with Path(config_path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    config = AppConfig.model_validate(raw)

    mlflow_uri = os.getenv("MLOPS_MLFLOW_URI")
    if mlflow_uri:
        config.tracking.mlflow_uri = mlflow_uri

    log_level = os.getenv("MLOPS_LOG_LEVEL")
    if log_level:
        config.monitoring.log_level = log_level

    enable_prometheus = os.getenv("MLOPS_ENABLE_PROMETHEUS")
    if enable_prometheus is not None:
        config.monitoring.enable_prometheus = _parse_bool(enable_prometheus)

    return config
