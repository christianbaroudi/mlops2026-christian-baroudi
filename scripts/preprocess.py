"""Clean a raw listings file.

    uv run python scripts/preprocess.py --input data/drop-1/listings.csv \
        --output data/processed/drop-1.csv
"""

import argparse
import logging

from rental_pricing.preprocess import Preprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean a raw listings CSV file.")
    parser.add_argument("--input", required=True, help="raw listings CSV")
    parser.add_argument("--output", required=True, help="where to write the clean CSV")
    parser.add_argument(
        "--keep-all-rows",
        action="store_true",
        help="keep duplicates and rows without a price (scoring mode)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Preprocessor(args.input, args.output, training=not args.keep_all_rows).run()


if __name__ == "__main__":
    main()
