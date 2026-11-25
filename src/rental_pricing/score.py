"""Batch scoring stage: predict a price for every row of a raw listings file.

The model contains the cleaning and features steps, so the input file is used as it is.
The output has one row per input row (duplicates and dirty rows included): row_id,prediction.
"""

import logging
import time
from pathlib import Path

import mlflow
import pandas as pd

from rental_pricing import schema
from rental_pricing.base import Stage
from rental_pricing.prediction_log import PredictionLogger, model_info
from rental_pricing.preprocess import read_listings
from rental_pricing.train import MODEL_NAME

log = logging.getLogger(__name__)

DEFAULT_MODEL_URI = f"models:/{MODEL_NAME}@champion"


class BatchScorer(Stage):
    """Load a model, score a whole CSV file, write the predictions and log them."""

    name = "batch_scoring"

    def __init__(self, input_path, output_path, model_uri=DEFAULT_MODEL_URI):
        self.input_path = input_path
        self.output_path = output_path
        self.model_uri = model_uri

    def run(self) -> pd.DataFrame:
        start = time.perf_counter()

        model = mlflow.pyfunc.load_model(self.model_uri)
        listings = read_listings(self.input_path)
        log.info("Scoring %d rows from %s with %s", len(listings), self.input_path, self.model_uri)

        # A file with prices (e.g. a drop) can be scored too: the price is simply ignored.
        predictions = model.predict(listings.drop(columns=schema.TARGET, errors="ignore"))
        output = pd.DataFrame({schema.ROW_ID: listings[schema.ROW_ID], "prediction": predictions})

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(self.output_path, index=False)
        log.info("Wrote %d predictions to %s", len(output), self.output_path)

        # latency_ms = time of the whole job / number of rows.
        latency_ms = (time.perf_counter() - start) * 1000 / len(listings)
        inputs = listings.to_json(orient="records", lines=True).splitlines()
        logger = PredictionLogger(model_info(self.model_uri, MODEL_NAME))
        logger.log(inputs, predictions, latency_ms)
        return output
