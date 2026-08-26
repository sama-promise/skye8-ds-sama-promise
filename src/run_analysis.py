import argparse
import logging
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    raw_dir: Path
    figures_dir: Path
    report_dir: Path
    date_cutoff: str = "2026-08-25"
    max_acceptable_queue_min: int = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the rural water programme analysis."
    )
    parser.add_argument("--raw", required=True, help="Path to raw data folder")
    parser.add_argument("--figures", required=True, help="Path to save figures")
    parser.add_argument("--report", required=True, help="Path to save reports")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        raw_dir=Path(args.raw),
        figures_dir=Path(args.figures),
        report_dir=Path(args.report),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting analysis script...")

    args = parse_args()
    config = build_config(args)
    logger.info("Config loaded: %s", config)


if __name__ == "__main__":
    main()