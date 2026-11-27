"""Move champion to the challenger if it has a lower MAE on the drop 2 holdout.

uv run python scripts/promote.py --drop-1 data/drop-1/listings.csv \
    --drop-2 data/drop-2/listings.csv
"""

import argparse
import logging

from rental_pricing.promote import Promoter
from rental_pricing.train import MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote the challenger if it is better.")
    parser.add_argument("--drop-1", required=True, help="raw drop 1 CSV (to find new listings)")
    parser.add_argument(
        "--drop-2", required=True, help="raw drop 2 CSV (the holdout comes from it)"
    )
    parser.add_argument("--model-name", default=MODEL_NAME, help="registered model name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Promoter(args.drop_1, args.drop_2, model_name=args.model_name).run()


if __name__ == "__main__":
    main()
