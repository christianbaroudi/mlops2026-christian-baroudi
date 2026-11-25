"""Online service: answer one listing at a time.

POST /predict with one listing as JSON (the columns of listings.csv, raw values are fine,
missing fields are allowed) returns {"prediction": <price in credits>}.

The model is loaded once, when the service starts. One start of the service is one run
of the prediction log.
"""

import json
import logging
import time
from typing import Any

import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI

from rental_pricing.base import Stage
from rental_pricing.prediction_log import PredictionLogger, model_info
from rental_pricing.score import DEFAULT_MODEL_URI
from rental_pricing.train import MODEL_NAME

log = logging.getLogger(__name__)


def create_app(model_uri: str) -> FastAPI:
    """Load the model and build the web app around it."""
    model = mlflow.pyfunc.load_model(model_uri)
    logger = PredictionLogger(model_info(model_uri, MODEL_NAME))
    log.info("Loaded %s", model_uri)

    app = FastAPI(title="Rental pricing")

    @app.post("/predict")
    def predict(listing: dict[str, Any]) -> dict:
        start = time.perf_counter()
        prediction = float(model.predict(pd.DataFrame([listing]))[0])
        latency_ms = (time.perf_counter() - start) * 1000

        logger.log([json.dumps(listing)], [prediction], latency_ms)
        return {"prediction": prediction}

    return app


class PredictionService(Stage):
    """Start the web service on the given port."""

    name = "service"

    def __init__(self, model_uri=DEFAULT_MODEL_URI, host="0.0.0.0", port=8000):
        self.model_uri = model_uri
        self.host = host
        self.port = port

    def run(self) -> None:
        uvicorn.run(create_app(self.model_uri), host=self.host, port=self.port)
