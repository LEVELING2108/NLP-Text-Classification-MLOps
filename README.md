# NLP Text Classification MLOps Project

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.19.0-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-F46800?logo=grafana&logoColor=white)
![SQLite](https://img.shields.io/badge/Tracking%20DB-SQLite-003B57?logo=sqlite&logoColor=white)

![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)
![Testing](https://img.shields.io/badge/Tests-Pytest-0A9EDC?logo=pytest&logoColor=white)
![API Docs](https://img.shields.io/badge/API-Swagger%20OpenAPI-85EA2D?logo=swagger&logoColor=black)
![Model Versioning](https://img.shields.io/badge/Model%20Versioning-Enabled-4CAF50)
![Experiment Tracking](https://img.shields.io/badge/Experiment%20Tracking-Enabled-4CAF50)
![Monitoring](https://img.shields.io/badge/Monitoring-Basic%20Setup-4CAF50)
![Status](https://img.shields.io/badge/MLOps-Production%20Grade-6A1B9A)

<!-- Optional live workflow badge (replace placeholders):
[![CI-CD](https://github.com/<your-username>/<your-repo>/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions/workflows/ci-cd.yml)
-->

Production-grade end-to-end MLOps project for NLP text classification (spam detection) with training, tracking, serving, CI/CD, containerization, and monitoring hooks.

## 1. Architecture (Text Diagram)

```text
                +-----------------------+
                |  data/raw/*.csv       |
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                |  Training Pipeline    |
                |  - ingestion          |
                |  - preprocessing      |
                |  - TF-IDF             |
                |  - LogisticRegression |
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                | MLflow Tracking       |
                | (sqlite://mlflow.db)  |
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                | models/<version>/     |
                | model + vectorizer +  |
                | metadata + latest.json|
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                | FastAPI Inference API |
                | /predict /health      |
                | /metrics (Prometheus) |
                +-----------+-----------+
                            |
            +---------------+-----------------+
            |                                 |
            v                                 v
 +-----------------------+         +----------------------+
 | Prometheus            |         | Grafana              |
 | scrape /metrics       |         | dashboards           |
 +-----------------------+         +----------------------+

CI/CD (GitHub Actions):
tests -> train -> docker build -> local smoke deploy
```

## 2. Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── api/
│   └── main.py
├── configs/
│   └── config.yaml
├── data/
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── spam_dataset.csv
├── deployment/
│   ├── docker-compose.yml
│   ├── grafana/
│   │   └── provisioning/
│   │       └── datasources/
│   │           └── datasource.yml
│   └── prometheus/
│       └── prometheus.yml
├── models/
│   └── .gitkeep
├── notebooks/
│   └── README.md
├── pipelines/
│   └── run_train.py
├── src/
│   └── mlops_nlp/
│       ├── data/
│       │   ├── ingestion.py
│       │   └── preprocessing.py
│       ├── features/
│       │   └── vectorizer.py
│       ├── models/
│       │   ├── evaluate.py
│       │   └── train.py
│       ├── pipelines/
│       │   ├── inference_pipeline.py
│       │   └── train_pipeline.py
│       ├── tracking/
│       │   └── mlflow_utils.py
│       ├── utils/
│       │   └── io.py
│       ├── config.py
│       ├── logging_config.py
│       └── schemas.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_preprocessing.py
│   └── test_train_pipeline.py
├── Dockerfile
├── requirements-dev.txt
├── pytest.ini
├── requirements.txt
└── README.md
```

## 3. Tech Stack

- Python 3.11+
- Scikit-learn
- FastAPI + Pydantic
- MLflow (experiment tracking + local SQLite backend)
- Docker + Docker Compose
- GitHub Actions (CI/CD)
- Prometheus + Grafana (monitoring basics)

## 4. Setup & Run

### 4.1 Local Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

For production/runtime-only installs (for example inside Docker), use:
```bash
pip install -r requirements.txt
```

### 4.2 Train the Model

```bash
python pipelines/run_train.py --config configs/config.yaml
```

Artifacts are saved under `models/<version>/` and pointer file `models/latest.json`.

### 4.3 Run API Locally

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4.4 Run with Docker

```bash
docker build -t nlp-mlops-api:latest .
docker run -p 8000:8000 nlp-mlops-api:latest
```

Note: the Docker image runs one training step during build so deployed containers
start with ready model artifacts (`models/latest.json` + versioned bundle).

### 4.5 Run Full Stack (API + Prometheus + Grafana)

```bash
cd deployment
docker compose up --build
```

- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (default `admin/admin`)

### 4.6 Common Commands (Makefile)

```bash
make install-dev
make test
make train
make serve
```

### 4.7 Environment Overrides

You can override key runtime settings without editing `configs/config.yaml`:

- `MLOPS_MLFLOW_URI` (example: `sqlite:///mlflow.db` or remote tracking URI)
- `MLOPS_LOG_LEVEL` (example: `DEBUG`, `INFO`, `WARNING`)
- `MLOPS_ENABLE_PROMETHEUS` (`true`/`false`, `1`/`0`, `yes`/`no`)

## 5. API Endpoints

### `GET /health`
- Returns service health and model load status.

### `POST /predict`
- Input (Pydantic validated):
```json
{
  "text": "Congratulations, you won free cash!"
}
```
- Output:
```json
{
  "prediction": "spam",
  "model_version": "20260412010101"
}
```

### `GET /metrics`
- Prometheus metrics endpoint.
- Includes request count, latency histogram, and prediction counters.

## 6. CI/CD (GitHub Actions)

Workflow file: `.github/workflows/ci-cd.yml`

Pipeline stages:
1. Checkout + Python setup
2. Install dependencies
3. Run tests (`pytest`)
4. Train model (`python pipelines/run_train.py`)
5. Build Docker image
6. Run container + `/health` smoke test

## 7. Testing

Run unit tests:

```bash
python -m pytest -q
```

Included tests:
- Text preprocessing unit tests
- Training pipeline integration-style test
- FastAPI endpoint and validation tests

## 8. Monitoring Notes

- Structured logging configured via `src/mlops_nlp/logging_config.py`.
- Prometheus metrics exported from `api/main.py`.
- Grafana datasource provisioning file included.
- Dashboard JSON can be added later under `deployment/grafana/provisioning/`.

## 9. Optional Cloud Deployment

Recommended first target: **Render** for quick managed deployment.

### Render
- Create a Web Service from this repo.
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- Optional env vars:
  - `MLOPS_MLFLOW_URI`
  - `MLOPS_LOG_LEVEL`
  - `MLOPS_ENABLE_PROMETHEUS`

If using Docker deploy on Render, no extra training step is needed at runtime:
the image build already produces model artifacts.

### AWS EC2
- Install Docker and Docker Compose on instance.
- Clone repo and run `docker compose -f deployment/docker-compose.yml up --build -d`.
- Expose ports `8000`, `9090`, and `3000` via Security Group.

## 10. Screenshots Placeholders

- `[PLACEHOLDER]` MLflow experiment UI screenshot
- `[PLACEHOLDER]` FastAPI `/docs` screenshot
- `[PLACEHOLDER]` Prometheus targets screenshot
- `[PLACEHOLDER]` Grafana dashboard screenshot
