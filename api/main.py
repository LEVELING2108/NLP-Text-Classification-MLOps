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

LOGGER = get_logger(__name__)
REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["path", "method", "status"])
PREDICTION_COUNT = Counter("prediction_total", "Total inference predictions", ["prediction"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "HTTP request latency", ["path", "method"])

pipeline: InferencePipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    config = load_config()
    configure_logging(config.monitoring.log_level)
    try:
        pipeline = InferencePipeline()
        LOGGER.info("Loaded inference pipeline successfully")
    except FileNotFoundError:
        pipeline = None
        LOGGER.warning("No trained model artifacts detected. /predict will fail until model is trained.")
    yield


app = FastAPI(
    title=load_config().api.title,
    version=load_config().api.version,
    description=load_config().api.description,
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = perf_counter()
    response = await call_next(request)
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
    prediction = pipeline.predict(payload.text)
    PREDICTION_COUNT.labels(prediction=prediction).inc()
    return PredictionResponse(prediction=prediction, model_version=pipeline.metadata["model_version"])


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

