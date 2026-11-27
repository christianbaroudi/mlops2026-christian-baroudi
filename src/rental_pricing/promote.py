"""Promote stage: decide whether the challenger replaces the champion.

1. The holdout: the rows of drop 2 whose listing_key is not in drop 1 (new listings, like
   the hidden set), with a valid price. They are used raw, as the hidden set will be.
2. Both models (aliases champion and challenger) predict the holdout: MAE in credits.
3. If the challenger's MAE is lower, the alias champion moves to the challenger's version.
   Otherwise nothing changes. The MAEs and the decision are logged as one MLflow run.
"""

import logging

import mlflow
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from rental_pricing import schema
from rental_pricing.base import Stage
from rental_pricing.preprocess import clean_listings, is_new_listing, read_listings
from rental_pricing.train import EXPERIMENT, MODEL_NAME

log = logging.getLogger(__name__)


class Promoter(Stage):
    """Compare champion and challenger on the drop 2 holdout and move champion if needed."""

    name = "promote"

    def __init__(self, drop_1_path, drop_2_path, model_name=MODEL_NAME):
        self.drop_1_path = drop_1_path
        self.drop_2_path = drop_2_path
        self.model_name = model_name

    def run(self) -> bool:
        drop_1 = read_listings(self.drop_1_path)
        drop_2 = read_listings(self.drop_2_path)
        prices = clean_listings(drop_2)[schema.TARGET]
        in_holdout = is_new_listing(drop_2, drop_1) & prices.notna()
        holdout = drop_2[in_holdout].drop(columns=schema.TARGET)
        true_prices = prices[in_holdout]
        log.info("Holdout: %d rows of listings that are new in drop 2", len(holdout))

        client = MlflowClient()
        versions, maes = {}, {}
        for alias in ["champion", "challenger"]:
            versions[alias] = client.get_model_version_by_alias(self.model_name, alias).version
            model = mlflow.pyfunc.load_model(f"models:/{self.model_name}@{alias}")
            maes[alias] = mean_absolute_error(true_prices, model.predict(holdout))
            log.info("%s (version %s): MAE = %.2f credits", alias, versions[alias], maes[alias])

        promoted = maes["challenger"] < maes["champion"]

        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name="promote"):
            mlflow.log_params(
                {
                    "champion_version": versions["champion"],
                    "challenger_version": versions["challenger"],
                    "holdout_rows": len(holdout),
                }
            )
            mlflow.log_metrics(
                {"champion_mae": maes["champion"], "challenger_mae": maes["challenger"]}
            )
            mlflow.set_tag("promoted", str(promoted))

        if promoted:
            client.set_registered_model_alias(self.model_name, "champion", versions["challenger"])
            log.info("Promoted: champion is now version %s", versions["challenger"])
        else:
            log.info("Not promoted: champion stays version %s", versions["champion"])
        return promoted
