"""Train stage: train every model family, log each run to MLflow, register the best one.

1. Split the cleaned training file: 80% to train, 20% to validate. A listing_key is
   never on both sides, so validation listings are new to the model (like the hidden set).
2. For each model family: train, compute the validation MAE, log everything to MLflow,
   then retrain on all rows and log that model.
3. Register the family with the lowest validation MAE as a new version of the model
   name, and give it the alias (champion for drop 1, challenger for drop 2).
"""

import logging
import os
import subprocess

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from rental_pricing import schema
from rental_pricing.base import Stage
from rental_pricing.model import MODEL_FAMILIES, build_model
from rental_pricing.preprocess import read_listings

log = logging.getLogger(__name__)

MODEL_NAME = "rental-pricing"
EXPERIMENT = "rental-pricing"


def git_commit() -> str:
    """The current commit, from git or (inside Docker, where there is no .git) GIT_COMMIT."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return os.environ.get("GIT_COMMIT", "unknown")


class Trainer(Stage):
    """Train all model families on one cleaned file and register the best as `alias`."""

    name = "train"

    def __init__(self, data_path, landmarks_path, alias="champion", model_name=MODEL_NAME):
        self.data_path = data_path
        self.landmarks_path = landmarks_path
        self.alias = alias
        self.model_name = model_name

    def run(self) -> dict:
        data = read_listings(self.data_path)
        landmarks = pd.read_csv(self.landmarks_path)
        y = pd.to_numeric(data[schema.TARGET])
        X = data.drop(columns=schema.TARGET)

        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_rows, val_rows = next(split.split(X, y, groups=X[schema.LISTING_KEY]))
        log.info("Training on %d rows, validating on %d rows", len(train_rows), len(val_rows))

        mlflow.set_experiment(EXPERIMENT)
        results = []
        for family in MODEL_FAMILIES:
            with mlflow.start_run(run_name=family):
                model = build_model(family, landmarks)
                model.fit(X.iloc[train_rows], y.iloc[train_rows])
                val_mae = mean_absolute_error(y.iloc[val_rows], model.predict(X.iloc[val_rows]))

                mlflow.set_tag("git_commit", git_commit())
                mlflow.log_params({"model_family": family, "data": str(self.data_path)})
                mlflow.log_params(model.regressor["regressor"].get_params())
                mlflow.log_metric("val_mae", val_mae)

                # The logged model is retrained on every row, so it learns from all the data.
                model.fit(X, y)
                # cloudpickle, because the default format (skops) can't save our own
                # ListingFeatures step. Loading needs the rental_pricing package installed.
                logged = mlflow.sklearn.log_model(
                    model, name="model", serialization_format="cloudpickle"
                )
                log.info("%s: validation MAE = %.2f credits", family, val_mae)
                results.append({"family": family, "val_mae": val_mae, "uri": logged.model_uri})

        best = min(results, key=lambda result: result["val_mae"])
        version = mlflow.register_model(best["uri"], self.model_name)
        MlflowClient().set_registered_model_alias(self.model_name, self.alias, version.version)
        log.info(
            "Registered %s version %s (%s) with alias %s",
            self.model_name,
            version.version,
            best["family"],
            self.alias,
        )
        return best
