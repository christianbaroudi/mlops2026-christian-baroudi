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

### Lint

```bash
uv run ruff check .
uv run ruff format --check .
```

## CI/CD

- **CI** (`.github/workflows/ci.yml`), on every pull request: `uv sync --locked`, the lint
  check, and a run of preprocessing on the sample in `tests/data/`.

## Metric

_To be completed: the metric optimised during training, and why._
