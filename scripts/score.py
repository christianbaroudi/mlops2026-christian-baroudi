"""Score a whole listings file with a model.

    uv run python scripts/score.py --input data/drop-1/listings.csv \
        --output data/predictions/drop-1.csv
"""

import argparse
import logging

from rental_pricing.score import DEFAULT_MODEL_URI, BatchScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a listings CSV file.")
    parser.add_argument("--input", required=True, help="listings CSV to score (raw is fine)")
    parser.add_argument("--output", required=True, help="where to write row_id,prediction")
    parser.add_argument(
        "--model-uri",
        default=DEFAULT_MODEL_URI,
        help=f"model to load, e.g. release/model (default: {DEFAULT_MODEL_URI})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    BatchScorer(args.input, args.output, args.model_uri).run()


if __name__ == "__main__":
    main()
