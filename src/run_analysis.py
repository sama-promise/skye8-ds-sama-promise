import argparse
import logging
from pathlib import Path
from dataclasses import dataclass


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting analysis script...")


if __name__ == "__main__":
    main()