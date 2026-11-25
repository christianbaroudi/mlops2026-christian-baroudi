"""The model: cleaning + features + encoding + a regressor, in one scikit-learn pipeline.

Because cleaning and features are inside the pipeline, the saved model takes a raw,
dirty listings table and returns prices. Batch scoring and the service only call
model.predict().

Every model family goes through the same build_model() function: only the regressor
at the end changes.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rental_pricing import schema
from rental_pricing.features import build_features
from rental_pricing.preprocess import clean_listings

# The model families. Each one is a regressor that receives the same encoded columns.
MODEL_FAMILIES = {
    "ridge": lambda: Ridge(alpha=10.0),
    "gradient_boosting": lambda: HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.05, max_iter=600, random_state=42
    ),
}


class ListingFeatures(BaseEstimator, TransformerMixin):
    """Raw listings -> cleaned listings with features (the preprocess and features stages).

    The landmarks table is stored inside, so the saved model does not need landmarks.csv.
    """

    def __init__(self, landmarks: pd.DataFrame):
        self.landmarks = landmarks

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # A missing column (for example in a JSON request) becomes a column of missing values.
        df = pd.DataFrame(X).reindex(columns=schema.INPUT_COLUMNS)
        df = build_features(clean_listings(df), self.landmarks)
        not_features = [schema.ROW_ID, schema.LISTING_KEY] + schema.DATES + schema.TEXT
        return df.drop(columns=not_features)


def encoder() -> ColumnTransformer:
    """Categories -> one-hot columns. Numbers -> missing values filled, then scaled."""
    categories = Pipeline(
        [
            ("fill", SimpleImputer(strategy="constant", fill_value="missing")),
            # Rare and unseen categories share one "infrequent" column.
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False
                ),
            ),
        ]
    )
    numbers = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("categories", categories, schema.CATEGORICAL),
            ("numbers", numbers, make_column_selector(dtype_include="number")),
        ]
    )


def build_model(family: str, landmarks: pd.DataFrame) -> TransformedTargetRegressor:
    """The full model for one family. It learns log(price) and predicts in credits."""
    pipeline = Pipeline(
        [
            ("features", ListingFeatures(landmarks)),
            ("encode", encoder()),
            ("regressor", MODEL_FAMILIES[family]()),
        ]
    )
    # Prices are very skewed (16 to 2549 credits): learning the log makes errors on cheap
    # and expensive listings count alike; np.exp turns predictions back into credits.
    return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)
