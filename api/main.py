from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from mlops_nlp.config import load_config
from mlops_nlp.logging_config import configure_logging, get_logger
from mlops_nlp.pipelines.inference_pipeline import InferencePipeline
from mlops_nlp.schemas import PredictionRequest, PredictionResponse
from mlops_nlp.utils.drift import log_inference

APP_CONFIG = load_config()
LOGGER = get_logger(__name__)
REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["path", "method", "status"])
PREDICTION_COUNT = Counter("prediction_total", "Total inference predictions", ["prediction"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "HTTP request latency", ["path", "method"])
PROMETHEUS_ENABLED = APP_CONFIG.monitoring.enable_prometheus

pipeline: InferencePipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    configure_logging(APP_CONFIG.monitoring.log_level)
    try:
        pipeline = InferencePipeline()
        LOGGER.info("Loaded inference pipeline successfully")
    except FileNotFoundError:
        pipeline = None
        LOGGER.warning("No trained model artifacts detected. /predict will fail until model is trained.")
    yield


app = FastAPI(
    title=APP_CONFIG.api.title,
    version=APP_CONFIG.api.version,
    description=APP_CONFIG.api.description,
    lifespan=lifespan,
)


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": APP_CONFIG.api.title,
            "version": APP_CONFIG.api.version,
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics" if PROMETHEUS_ENABLED else "disabled",
        }
    )


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = perf_counter()
    response = await call_next(request)
    if not PROMETHEUS_ENABLED:
        return response
    latency = perf_counter() - start

    path = request.url.path
    method = request.method
    status = str(response.status_code)
    REQUEST_COUNT.labels(path=path, method=method, status=status).inc()
    REQUEST_LATENCY.labels(path=path, method=method).observe(latency)
    return response


@app.get("/health")
def health() -> JSONResponse:
    status = "ok" if pipeline is not None else "degraded"
    return JSONResponse({"status": status})


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train model first.")
    prediction, confidence = pipeline.predict(payload.text)
    if PROMETHEUS_ENABLED:
        PREDICTION_COUNT.labels(prediction=prediction).inc()
    
    # Log for drift detection
    log_inference(
        log_path=APP_CONFIG.monitoring.inference_log_path,
        text=payload.text,
        prediction=prediction,
        confidence=confidence,
        model_version=pipeline.metadata["model_version"],
    )
    
    return PredictionResponse(
        prediction=prediction, 
        confidence=confidence,
        model_version=pipeline.metadata["model_version"]
    )


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    if not PROMETHEUS_ENABLED:
        raise HTTPException(status_code=404, detail="Prometheus metrics are disabled.")
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
