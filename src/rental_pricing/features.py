"""Features stage: turn cleaned listings into the columns the model learns from.

build_features() adds:
- the distance (km) from the listing to each landmark,
- the number of baths and whether the bath is shared, from bath_desc,
- one 0/1 column per amenity,
- durations instead of dates: host years, days since the first and last review.

It never removes rows, so it is also used when scoring.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rental_pricing import schema
from rental_pricing.base import Stage

log = logging.getLogger(__name__)

# Durations are measured from this fixed date (just after the last review in drop 1),
# so a listing gets the same features at training and at scoring time.
REFERENCE_DATE = pd.Timestamp("2025-10-01")


def landmark_distances(df: pd.DataFrame, landmarks: pd.DataFrame) -> pd.DataFrame:
    """One column per landmark: straight-line distance in km (positions are in metres)."""
    distances = {}
    for landmark in landmarks.itertuples():
        dx = df["pos_x"] - landmark.pos_x
        dy = df["pos_y"] - landmark.pos_y
        distances[f"dist_{landmark.name}_km"] = np.sqrt(dx**2 + dy**2) / 1000
    return pd.DataFrame(distances, index=df.index)


def bath_features(bath_desc: pd.Series) -> pd.DataFrame:
    """'1.5 shared baths' -> baths = 1.5, bath_shared = 1. 'half-bath' counts as 0.5."""
    text = bath_desc.astype("string").str.lower()
    baths = pd.to_numeric(text.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    baths = baths.mask(text.str.contains("half-bath", na=False), 0.5)
    shared = text.str.contains("shared").astype(float)  # missing text stays missing
    return pd.DataFrame({"baths": baths, "bath_shared": shared}, index=bath_desc.index)


def amenity_flags(amenities: pd.Series) -> pd.DataFrame:
    """One 0/1 column per known amenity, e.g. has_wifi."""
    lists = amenities.fillna("").str.split("|")
    flags = {}
    for amenity in schema.AMENITIES:
        column = "has_" + amenity.replace(" ", "_").replace("-", "_")
        flags[column] = lists.apply(lambda items, a=amenity: float(a in items))
    return pd.DataFrame(flags, index=amenities.index)


def date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dates become durations up to REFERENCE_DATE."""
    host_start = pd.to_datetime(df["host_start"], errors="coerce")
    first_review = pd.to_datetime(df["first_review"], errors="coerce")
    last_review = pd.to_datetime(df["last_review"], errors="coerce")
    return pd.DataFrame(
        {
            "host_years": (REFERENCE_DATE - host_start).dt.days / 365.25,
            "days_since_first_review": (REFERENCE_DATE - first_review).dt.days,
            "days_since_last_review": (REFERENCE_DATE - last_review).dt.days,
        },
        index=df.index,
    )


def build_features(df: pd.DataFrame, landmarks: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned table with the new feature columns (same rows, same order)."""
    new_columns = [
        landmark_distances(df, landmarks),
        bath_features(df["bath_desc"]),
        amenity_flags(df["amenities"]),
        date_features(df),
    ]
    return pd.concat([df] + new_columns, axis=1)


class FeatureBuilder(Stage):
    """Read a cleaned listings file, add the features and write the result."""

    name = "features"

    def __init__(self, input_path, landmarks_path, output_path):
        self.input_path = input_path
        self.landmarks_path = landmarks_path
        self.output_path = output_path

    def run(self) -> pd.DataFrame:
        df = pd.read_csv(self.input_path)
        landmarks = pd.read_csv(self.landmarks_path)
        log.info("Read %d rows and %d landmarks", len(df), len(landmarks))

        features = build_features(df, landmarks)

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(self.output_path, index=False)
        log.info("Wrote %d rows and %d columns to %s", *features.shape, self.output_path)
        return features
