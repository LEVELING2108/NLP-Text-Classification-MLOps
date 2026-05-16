# Monitoring and Drift Detection

This project includes a comprehensive monitoring stack using Prometheus, Grafana, and custom data logging.

## 1. Metrics Stack

- **Prometheus:** Scrapes metrics from the FastAPI `/metrics` endpoint.
- **Grafana:** Visualizes metrics from Prometheus.
- **Metrics Tracked:**
    - `api_requests_total`: Total requests by path, method, and status.
    - `request_latency_seconds`: Histogram of request latency.
    - `prediction_total`: Count of predictions by label (spam/ham).

## 2. Drift Detection Data

The API logs every inference request to a structured JSONL file for future drift detection and model performance analysis.

- **Log Path:** `data/logs/inference.jsonl` (configurable in `config.yaml`)
- **Format:**
  ```json
  {
    "timestamp": "2026-05-14T12:00:00Z",
    "text": "sample message",
    "prediction": "ham",
    "confidence": 0.98,
    "model_version": "20260514120000"
  }
  ```

### How to use this data:
1.  **Analyze Confidence:** A drop in average confidence scores over time can indicate data drift.
2.  **Label Distribution:** Monitor if the ratio of `spam` to `ham` predictions changes significantly.
3.  **Retraining:** Use these logs to identify difficult samples and add them to your training dataset.

## 3. Grafana Setup

1.  Start the stack: `docker compose up -d`
2.  Access Grafana at `http://localhost:3000` (admin/admin).
3.  The Prometheus datasource is automatically provisioned.
4.  You can import or create dashboards to visualize the `prediction_total` counter to track model performance.
