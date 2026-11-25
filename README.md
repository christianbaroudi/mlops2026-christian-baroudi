# mlops2026-christian-baroudi

Suggested nightly price for new rental listings: a complete ML system, from raw data
to a served model.

Author: Christian Baroudi

## Project structure

```
src/rental_pricing/   # the package: one class per pipeline stage, behind one shared interface
scripts/              # one CLI script per job; they only parse arguments and call the classes
tests/data/           # a tiny sample of the data, used by CI
Dockerfile
compose.yaml
pyproject.toml
uv.lock
```

The data drops are not committed. Put them here locally (`data/` is git-ignored):

```
data/
├── drop-1/listings.csv
├── drop-2/listings.csv
└── landmarks.csv
```

## Setup

```bash
uv sync
```

Without the Compose stack, point MLflow at a local file:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db        # macOS / Linux
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"      # Windows PowerShell
```

## Running the jobs

Every stage is a class behind one shared interface (`rental_pricing.base.Stage`, with a
`run()` method). The scripts in `scripts/` only parse arguments and call these classes.

### 1. Preprocess

Cleans a raw listings file: unifies category spellings, turns impossible values
(e.g. `guests = 0`, `bedrooms = -1`, `min_nights = 9999`) into missing values, parses
dates and `t`/`f` flags. For training it also drops duplicate rows and rows without a
valid price; `--keep-all-rows` keeps every row (scoring mode).

```bash
uv run python scripts/preprocess.py --input data/drop-1/listings.csv --output data/processed/drop-1.csv
```

### 2. Features

Adds the columns the model learns from: the distance (km) to each landmark, the number
of baths and a shared-bath flag (from `bath_desc`), one 0/1 column per amenity, and
durations instead of dates (host years, days since the first and last review, measured
from a fixed reference date). Rows are never removed.

```bash
uv run python scripts/features.py --input data/processed/drop-1.csv --landmarks data/landmarks.csv --output data/features/drop-1.csv
```

### 3. Train

Trains two model families through the same `build_model()` function: a linear model
(`ridge`) and gradient boosting (`gradient_boosting`). Each family is one MLflow run
(parameters, validation MAE, git commit, model). The one with the lowest validation MAE
is registered as a new version of the model `rental-pricing` and gets the alias given by
`--alias` (`champion` for drop 1, `challenger` for drop 2).

The saved model is one scikit-learn pipeline: cleaning → features → encoding → regressor.
So it takes the raw, dirty listings file directly, and scoring uses exactly the same code
as training.

```bash
uv run python scripts/train.py --input data/processed/drop-1.csv --landmarks data/landmarks.csv --alias champion
```

Validation MAE on drop 1 (20% of the listings, never seen in training):

| Model | MAE (credits) |
| --- | --- |
| Median price of the training rows (baseline) | 60.7 |
| ridge | 41.0 |
| gradient_boosting | **36.7** |

### Lint

```bash
uv run ruff check .
uv run ruff format --check .
```

## CI/CD

- **CI** (`.github/workflows/ci.yml`), on every pull request: `uv sync --locked`, the lint
  check, and a run of preprocessing and features on the sample in `tests/data/`.

## Metric

The model is graded on the **MAE in credits**, so that is what we optimise and compare:

- The models learn **log(price)**. Prices go from 16 to 2549 credits and are very skewed;
  on the log scale a few very expensive listings don't dominate training.
- Gradient boosting uses the **absolute error loss**. On the log scale it predicts the
  median price for similar listings, and the median is the value that minimises the MAE.
- Model selection and promotion use the **MAE in credits** on listings the model has
  never seen (split by `listing_key`), because the hidden set is made of new listings.
