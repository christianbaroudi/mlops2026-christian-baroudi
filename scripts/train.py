"""Train every model family and register the best one in MLflow.

    uv run python scripts/train.py --input data/processed/drop-1.csv \
        --landmarks data/landmarks.csv --alias champion
"""

import argparse
import logging

from rental_pricing.train import MODEL_NAME, Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the models and register the best one.")
    parser.add_argument("--input", required=True, help="cleaned training CSV (from preprocess)")
    parser.add_argument("--landmarks", required=True, help="landmarks CSV")
    parser.add_argument(
        "--alias", default="champion", help="alias of the new version: champion or challenger"
    )
    parser.add_argument("--model-name", default=MODEL_NAME, help="registered model name")
    parser.add_argument(
        "--exclude-new-vs",
        help="drop 1 CSV: leave out the listings that are not in it (the promotion holdout)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Trainer(
        args.input,
        args.landmarks,
        alias=args.alias,
        model_name=args.model_name,
        exclude_new_vs=args.exclude_new_vs,
    ).run()


if __name__ == "__main__":
    main()
