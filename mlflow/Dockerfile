# From the Experiment Tracking lab.
# The official MLflow image has no Postgres driver: without one, the server
# fails at startup with a postgresql:// backend store. The image's
# SQLAlchemy (2.1) opens postgresql:// URIs with psycopg 3, not psycopg2.
# Keep this tag equal to the mlflow version in pyproject.toml.
FROM ghcr.io/mlflow/mlflow:v3.18.0
RUN pip install --no-cache-dir "psycopg[binary]"
