import argparse
import logging
from pathlib import Path
from dataclasses import dataclass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the rural water programme analysis."
    )
    parser.add_argument("--raw", required=True, help="Path to raw data folder")
    parser.add_argument("--figures", required=True, help="Path to save figures")
    parser.add_argument("--report", required=True, help="Path to save reports")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting analysis script...")

    args = parse_args()
    logger.info("Raw data folder: %s", args.raw)
    logger.info("Figures output folder: %s", args.figures)
    logger.info("Report output folder: %s", args.report)


if __name__ == "__main__":
    main()