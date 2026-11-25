"""Start the online service (POST /predict on port 8000).

uv run python scripts/serve.py
"""

import argparse
import logging

from rental_pricing.score import DEFAULT_MODEL_URI
from rental_pricing.service import PredictionService


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the prediction service.")
    parser.add_argument(
        "--model-uri",
        default=DEFAULT_MODEL_URI,
        help=f"model to load, e.g. release/model (default: {DEFAULT_MODEL_URI})",
    )
    parser.add_argument("--port", type=int, default=8000, help="port to listen on")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    PredictionService(model_uri=args.model_uri, port=args.port).run()


if __name__ == "__main__":
    main()
