"""FastAPI inference service with JSON logging and Prometheus metrics."""
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("ml-api")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["path"])
PREDICTIONS = Counter("predictions_total", "Predictions by class", ["predicted_class"])

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(MODEL_PATH)
    state["pipeline"] = bundle["pipeline"]
    state["classes"] = bundle["classes"]
    logger.info("model loaded", extra={"extra_fields": {"path": MODEL_PATH}})
    yield
    state.clear()


app = FastAPI(title="Iris ML API", version="1.0.0", lifespan=lifespan)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=20)
    sepal_width: float = Field(..., gt=0, lt=20)
    petal_length: float = Field(..., gt=0, lt=20)
    petal_width: float = Field(..., gt=0, lt=20)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception:
        logger.exception("unhandled error", extra={"extra_fields": {"path": request.url.path}})
        raise
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(path).observe(elapsed)
        if path != "/metrics":
            logger.info(
                "request",
                extra={"extra_fields": {
                    "method": request.method, "path": path,
                    "status": status, "duration_ms": round(elapsed * 1000, 2),
                }},
            )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "pipeline" in state}


@app.post("/predict")
def predict(f: IrisFeatures):
    if "pipeline" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    x = np.array([[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]])
    proba = state["pipeline"].predict_proba(x)[0]
    idx = int(proba.argmax())
    label = state["classes"][idx]
    PREDICTIONS.labels(label).inc()
    return {
        "prediction": label,
        "confidence": round(float(proba[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(state["classes"], proba)},
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
