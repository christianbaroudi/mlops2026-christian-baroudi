"""Preprocess stage: clean a raw listings file.

clean_listings() fixes the values of each column. It never removes rows, so it can
also be used on the file to score, which needs one prediction per input row.

drop_training_rows() removes duplicates and rows without a price. Training only.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rental_pricing import schema
from rental_pricing.base import Stage

log = logging.getLogger(__name__)

# The room type is recognised by its first word: "ENTIRE HOME/APT", "entire home" -> "entire".
ROOM_TYPES = {
    "entire": "Entire home/apt",
    "private": "Private room",
    "shared": "Shared room",
    "hotel": "Hotel room",
}

# "t"/"f" in the raw file, 1.0/0.0 once cleaned.
BOOLEAN_VALUES = {"t": 1.0, "true": 1.0, "1.0": 1.0, "f": 0.0, "false": 0.0, "0.0": 0.0}


def read_listings(path) -> pd.DataFrame:
    """Read a listings CSV with every column as text. Cleaning converts the types."""
    return pd.read_csv(path, dtype=str)


def normalise_text(values: pd.Series) -> pd.Series:
    """Lower-case, strip and collapse spaces. Empty text becomes missing."""
    text = values.astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
    text = text.replace("", pd.NA)
    return text.astype(object).fillna(np.nan)


def clean_dates(values: pd.Series) -> pd.Series:
    """Raw files use dd/mm/yyyy. Cleaned files use yyyy-mm-dd."""
    text = values.astype("string").str.strip()
    raw_format = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    clean_format = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    return raw_format.fillna(clean_format).astype("datetime64[ns]")


def clean_number(values: pd.Series, column: str) -> pd.Series:
    """Convert to a number. Values outside the valid range of the column become missing."""
    numbers = pd.to_numeric(values, errors="coerce").astype(float)
    low, high = schema.VALID_RANGES.get(column, (None, None))
    invalid = pd.Series(False, index=numbers.index)
    if low is not None:
        invalid |= numbers < low
    if high is not None:
        invalid |= numbers > high
    if invalid.any():
        log.info("%s: %d impossible values set to missing", column, invalid.sum())
    return numbers.mask(invalid)


def clean_listings(raw: pd.DataFrame) -> pd.DataFrame:
    """Fix every column of a listings table. The number of rows does not change."""
    df = raw.copy()

    if "room" in df:
        first_word = normalise_text(df["room"]).str.split(" ").str[0]
        df["room"] = first_word.map(ROOM_TYPES)
    if "property" in df:
        # "entire rental" is a cut-off spelling of "entire rental unit".
        prop = normalise_text(df["property"]).replace("entire rental", "entire rental unit")
        df["property"] = prop.str.capitalize()
    if "area_code" in df:
        df["area_code"] = normalise_text(df["area_code"]).str.upper()
    for column in ["host_response", "bath_desc", "amenities"]:
        if column in df:
            df[column] = normalise_text(df[column])

    for column in schema.BOOLEAN:
        if column in df:
            df[column] = normalise_text(df[column]).map(BOOLEAN_VALUES)
    for column in schema.DATES:
        if column in df:
            df[column] = clean_dates(df[column])
    for column in schema.NUMERIC + [schema.TARGET]:
        if column in df:
            df[column] = clean_number(df[column], column)

    return df


def drop_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows (same listing, different row_id) and rows without a price."""
    duplicated = df.drop(columns=schema.ROW_ID).duplicated()
    no_price = df[schema.TARGET].isna()
    log.info("Dropping %d duplicates and %d rows without a price", duplicated.sum(), no_price.sum())
    return df[~duplicated & ~no_price].reset_index(drop=True)


def is_new_listing(df: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """True for the rows whose listing_key is not in the reference file (e.g. drop 1).

    In drop 2, these new listings are the promotion holdout: they look like the hidden set.
    """
    return ~df[schema.LISTING_KEY].isin(reference[schema.LISTING_KEY])


class Preprocessor(Stage):
    """Read a raw listings file, clean it and write the result.

    training=True also drops duplicates and rows without a price.
    training=False keeps every row (for scoring).
    """

    name = "preprocess"

    def __init__(self, input_path, output_path, training=True):
        self.input_path = input_path
        self.output_path = output_path
        self.training = training

    def run(self) -> pd.DataFrame:
        raw = read_listings(self.input_path)
        log.info("Read %d rows from %s", len(raw), self.input_path)

        df = clean_listings(raw)
        if self.training:
            df = drop_training_rows(df)

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        log.info("Wrote %d rows to %s", len(df), self.output_path)
        return df
