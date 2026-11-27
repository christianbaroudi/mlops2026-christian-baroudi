# mlops2026-christian-baroudi

Suggested nightly price for new rental listings: a complete ML system, from raw data
to a served model.

Author: Christian Baroudi

## Project structure

```
src/rental_pricing/   # the package: one class per pipeline stage, behind one shared interface
scripts/              # one CLI script per job; they only parse arguments and call the classes
tests/data/           # a tiny sample of the data, used by CI
listing.json          # an example listing for the online service
mlflow/Dockerfile     # the MLflow server image (official image + Postgres driver)
Dockerfile            # the image of our jobs and service
compose.yaml          # db, adminer, mlflow + our jobs and service
.env.example          # copy to .env: the Postgres user, password and database
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

## Running with Docker Compose (lab PC, PowerShell)

The stack is the one of the Experiment Tracking lab (`db`, `adminer` on
http://localhost:8081, `mlflow` on http://localhost:5001), plus one service per job and the
online service. They all use the image that CD pushes to
`ghcr.io/christianbaroudi/mlops2026-christian-baroudi`, and they see `./data`, `./release`
and `./logs` through bind mounts, so a new file or model needs no rebuild.

First time only:

```powershell
Copy-Item .env.example .env
docker compose up -d --wait
```

Then create the prediction log database once: in Adminer (System `PostgreSQL`, Server `db`,
user and password from `.env`), run `CREATE DATABASE monitoring;`. Or from PowerShell:

```powershell
docker compose exec db psql -U mlflow_user -d mlflow_db -c "CREATE DATABASE monitoring;"
```

The jobs (the arguments after the service name go to the script):

```powershell
docker compose run --rm preprocess --input data/drop-1/listings.csv --output data/processed/drop-1.csv
docker compose run --rm features --input data/processed/drop-1.csv --landmarks data/landmarks.csv --output data/features/drop-1.csv
docker compose run --rm train --input data/processed/drop-1.csv --landmarks data/landmarks.csv --alias champion
docker compose run --rm preprocess --input data/drop-2/listings.csv --output data/processed/drop-2.csv
docker compose run --rm train --input data/processed/drop-2.csv --landmarks data/landmarks.csv --alias challenger --exclude-new-vs data/drop-1/listings.csv
docker compose run --rm promote --drop-1 data/drop-1/listings.csv --drop-2 data/drop-2/listings.csv
docker compose run --rm score --input data/drop-1/listings.csv --output data/predictions/drop-1.csv
docker compose run --rm score --input data/drop-1/listings.csv --output data/predictions/drop-1.csv --model-uri release/model
```

The online service:

```powershell
docker compose up -d service
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '@listing.json'
```

To use a locally built image instead of the one from ghcr.io, add `--build`
(for example `docker compose run --rm --build score ...`).

## Running the jobs with uv

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

The challenger is trained on drop 2. `--exclude-new-vs` leaves out the listings that are
new in drop 2 (not in drop 1): they are the promotion holdout, so the challenger must
not see them.

```bash
uv run python scripts/preprocess.py --input data/drop-2/listings.csv --output data/processed/drop-2.csv
uv run python scripts/train.py --input data/processed/drop-2.csv --landmarks data/landmarks.csv --alias challenger --exclude-new-vs data/drop-1/listings.csv
```

### 4. Promote

Decides whether the challenger replaces the champion. The holdout is the rows of drop 2
whose `listing_key` is not in drop 1 (new listings, like the hidden set), used raw. Both
models predict it; if the challenger's MAE is lower, the alias `champion` moves to the
challenger's version, otherwise nothing changes. The two MAEs and the decision are logged
as an MLflow run named `promote`. Batch scoring and the service then load the new
champion with no code change.

```bash
uv run python scripts/promote.py --drop-1 data/drop-1/listings.csv --drop-2 data/drop-2/listings.csv
```

### 5. Batch scoring

Scores a whole CSV file, raw and dirty as it is. The output has the columns
`row_id,prediction`, with one row per input row (duplicates and dirty rows included).
`--model-uri` defaults to `models:/rental-pricing@champion`; it can also be a local
folder such as `release/model`.

```bash
uv run python scripts/score.py --input data/drop-1/listings.csv --output data/predictions/drop-1.csv
uv run python scripts/score.py --input data/drop-1/listings.csv --output data/predictions/drop-1.csv --model-uri release/model
```

### 6. Online service

A FastAPI service on port 8000. `POST /predict` takes one listing as JSON (the columns of
`listings.csv`; raw values and missing fields are fine) and returns its predicted price.
The model is loaded once, at start-up (default `models:/rental-pricing@champion`, or
`--model-uri release/model`).

```bash
uv run python scripts/serve.py
```

Then, from another terminal, with the example listing in `listing.json`:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @listing.json
```

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '@listing.json'
```

Answer: `{"prediction": 106.46}` (credits). The interactive docs are at
http://localhost:8000/docs.

### Prediction log

Batch scoring and the service log every prediction (`src/rental_pricing/prediction_log.py`)
with the columns `logged_at, inputs, prediction, model_name, model_version, model_alias,
latency_ms`. For batch scoring, `latency_ms` is the time of the whole job divided by the
number of rows. With a local folder as `--model-uri`, `model_version` and `model_alias`
are empty.

- In the Compose stack, `MONITORING_DB_URI` is set, and rows go to the table `predictions`
  of the database `monitoring` on the `db` service (created on the first insert).
- Without it (a laptop), rows go to `logs/<run-id>/predictions.csv`, one folder per run.

To check the last predictions in the stack (or the same query in Adminer, database
`monitoring`):

```powershell
docker compose exec db psql -U mlflow_user -d monitoring -c "SELECT logged_at, model_name, model_version, model_alias, prediction FROM predictions ORDER BY logged_at DESC LIMIT 5;"
```

### Lint

```bash
uv run ruff check .
uv run ruff format --check .
```

## CI/CD

- **CI** (`.github/workflows/ci.yml`), on every pull request: `uv sync --locked`, the lint
  check, and a run of preprocessing and features on the sample in `tests/data/`.
- **CD** (`.github/workflows/cd.yml`), on every merge to `master`: builds the image and
  pushes it to `ghcr.io/christianbaroudi/mlops2026-christian-baroudi`, tagged `latest`
  and with the commit hash.

## Releasing the model

The graded model is the one attached to the latest GitHub release. On a lab PC, with the
stack running, from the repository root (PowerShell):

1. Check the registry version of the champion (it goes in the tag):

   ```powershell
   $env:MLFLOW_TRACKING_URI = "http://localhost:5001"
   uv run python -c "from mlflow import MlflowClient; print(MlflowClient().get_model_version_by_alias('rental-pricing', 'champion').version)"
   ```

2. Export the champion (the first line deletes the previous export):

   ```powershell
   Remove-Item release -Recurse -Force -ErrorAction SilentlyContinue
   $env:MLFLOW_TRACKING_URI = "http://localhost:5001"
   uv run mlflow artifacts download --artifact-uri "models:/rental-pricing@champion" --dst-path release/model
   Compress-Archive release/model release/model.zip
   ```

3. On GitHub: **Releases** → **Draft a new release** → tag `model-v<version>` (the number
   from step 1, for example `model-v2`) → attach `release/model.zip` → **Publish release**.

## At the defence

1. Replace the local export with the released model, and check that
   `release/model/MLmodel` exists:

   ```powershell
   Remove-Item release/model -Recurse -Force -ErrorAction SilentlyContinue
   Expand-Archive $HOME\Downloads\model.zip -DestinationPath release -Force
   Test-Path release/model/MLmodel
   ```

2. Put the instructor's file in `data/` (here `data/hidden.csv`) and score it with the
   released model:

   ```powershell
   docker compose run --rm score --input data/hidden.csv --output data/predictions/hidden.csv --model-uri release/model
   ```

3. Score it once more with the default model (`models:/rental-pricing@champion`):

   ```powershell
   docker compose run --rm score --input data/hidden.csv --output data/predictions/hidden-champion.csv
   ```

4. Check in the prediction log that `model_version` equals the version of the release
   tag `model-v<version>`:

   ```powershell
   docker compose exec db psql -U mlflow_user -d monitoring -c "SELECT logged_at, model_name, model_version, model_alias FROM predictions ORDER BY logged_at DESC LIMIT 5;"
   ```

## Metric

The model is graded on the **MAE in credits**, so that is what we optimise and compare:

- The models learn **log(price)**. Prices go from 16 to 2549 credits and are very skewed;
  on the log scale a few very expensive listings don't dominate training.
- Gradient boosting uses the **absolute error loss**. On the log scale it predicts the
  median price for similar listings, and the median is the value that minimises the MAE.
- Model selection and promotion use the **MAE in credits** on listings the model has
  never seen (split by `listing_key`), because the hidden set is made of new listings.
