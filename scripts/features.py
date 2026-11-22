"""Add model features to a cleaned listings file."""

import argparse
import logging

from rental_pricing.features import FeatureBuilder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="cleaned listings CSV")
    parser.add_argument("--landmarks", required=True, help="landmarks CSV")
    parser.add_argument("--output", required=True, help="where to write the features CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    FeatureBuilder(args.input, args.landmarks, args.output).run()


if __name__ == "__main__":
    main()
