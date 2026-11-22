"""Column names and value rules of listings.csv (see the data dictionary in the brief)."""

TARGET = "nightly_price"
ROW_ID = "row_id"
LISTING_KEY = "listing_key"

CATEGORICAL = ["room", "property", "area_code", "host_response"]
BOOLEAN = ["host_superhost", "instant_book"]
DATES = ["host_start", "first_review", "last_review"]
TEXT = ["amenities", "bath_desc"]
NUMERIC = [
    "pos_x",
    "pos_y",
    "guests",
    "rating",
    "bedrooms",
    "beds",
    "reviews_total",
    "reviews_last_year",
    "min_nights",
    "max_nights",
    "avail_30",
    "avail_365",
    "host_listings",
    "rating_location",
    "rating_value",
    "rating_clean",
]

# Every amenity that appears in the "|"-separated amenities column.
AMENITIES = [
    "kitchen",
    "wifi",
    "smoke alarm",
    "hot water",
    "essentials",
    "dishes and silverware",
    "bed linens",
    "hair dryer",
    "cooking basics",
    "refrigerator",
    "microwave",
    "hangers",
    "iron",
    "heating",
    "tv",
    "hot water kettle",
    "washer",
    "dedicated workspace",
    "shampoo",
    "cleaning products",
    "wine glasses",
    "oven",
    "dining table",
    "baking sheet",
    "dishwasher",
    "freezer",
    "toaster",
    "shower gel",
    "coffee maker",
    "self check-in",
]

# Values outside these (inclusive) bounds are impossible and become missing.
# None means "no bound on that side".
VALID_RANGES = {
    "guests": (1, 30),
    "bedrooms": (0, 50),
    "beds": (0, 50),
    "rating": (1, 5),
    "rating_location": (1, 5),
    "rating_value": (1, 5),
    "rating_clean": (1, 5),
    "reviews_total": (0, None),
    "reviews_last_year": (0, None),
    # Hosts accept up to ~1125 nights (plus some noise in the data). 9999 is impossible.
    "min_nights": (1, 1500),
    "max_nights": (1, 1500),
    "avail_30": (0, 30),
    "avail_365": (0, 365),
    "host_listings": (1, None),
    TARGET: (0.01, None),
}
