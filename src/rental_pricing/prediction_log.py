"""Prediction log, shared by batch scoring and the online service.

Each prediction adds one row with: logged_at, inputs, prediction, model_name,
model_version, model_alias, latency_ms. Nothing is overwritten.

Where the rows go:
- MONITORING_DB_URI is set (the Compose stack): table `predictions` in the database
  `monitoring` on the db service. The table is created on the first insert.
- otherwise (a laptop without the stack): logs/<run-id>/predictions.csv, one folder per
  run, where a run is one batch job or one start of the service.
"""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from mlflow import MlflowClient
from sqlalchemy import create_engine

log = logging.getLogger(__name__)


def model_info(model_uri: str, default_name: str) -> dict:
    """Name, version and alias of the model behind a URI.

    models:/rental-pricing@champion -> the registry gives the version of the alias.
    models:/rental-pricing/3       -> version 3, no alias.
    release/model (a local folder) -> no version and no alias (NULL in the database).
    """
    if not model_uri.startswith("models:/"):
        return {"model_name": default_name, "model_version": None, "model_alias": None}

    reference = model_uri.removeprefix("models:/")
    if "@" in reference:
        name, alias = reference.split("@")
        version = MlflowClient().get_model_version_by_alias(name, alias).version
        return {"model_name": name, "model_version": str(version), "model_alias": alias}
    name, version = reference.split("/")
    return {"model_name": name, "model_version": version, "model_alias": None}


class PredictionLogger:
    """Appends predictions to the database table or to this run's CSV file."""

    def __init__(self, model: dict):
        self.model = model
        self.db_uri = os.environ.get("MONITORING_DB_URI")
        # A run ID is the time the run started, e.g. 20261125-174915-123456.
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        self.csv_path = Path("logs") / run_id / "predictions.csv"

    def log(self, inputs: list[str], predictions, latency_ms: float) -> None:
        """inputs: one JSON text per listing. predictions: one price per listing."""
        rows = pd.DataFrame(
            {
                "logged_at": datetime.now(UTC),
                "inputs": inputs,
                "prediction": predictions,
                "model_name": self.model["model_name"],
                "model_version": self.model["model_version"],
                "model_alias": self.model["model_alias"],
                "latency_ms": latency_ms,
            }
        )
        if self.db_uri:
            rows.to_sql("predictions", create_engine(self.db_uri), if_exists="append", index=False)
            log.info("Logged %d predictions to the predictions table", len(rows))
        else:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not self.csv_path.exists()
            rows.to_csv(self.csv_path, mode="a", header=new_file, index=False)
            log.info("Logged %d predictions to %s", len(rows), self.csv_path)
