import argparse
import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd


class MissingColumnError(Exception):
    def __init__(self, filename: str, column: str):
        self.filename = filename
        self.column = column
        super().__init__(f"{filename} is missing required column: {column}")


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


def validate_columns(df: pd.DataFrame, required: list[str], filename: str) -> None:
    for column in required:
        if column not in df.columns:
            raise MissingColumnError(filename, column)


def load_water_points(raw_dir: Path) -> pd.DataFrame:
    filename = "water_points.csv"
    df = pd.read_csv(raw_dir / filename)
    required = [
        "point_id", "village", "division", "point_type", "installed_on",
        "depth_m", "households_served", "managed_by", "latitude", "longitude",
    ]
    validate_columns(df, required, filename)
    return df


def load_inspections(raw_dir: Path) -> pd.DataFrame:
    filename = "inspections.csv"
    df = pd.read_csv(raw_dir / filename)
    required = [
        "inspection_id", "point_id", "inspected_on", "functional",
        "water_quality", "queue_minutes", "inspector_id",
    ]
    validate_columns(df, required, filename)
    return df


def load_repairs(raw_dir: Path) -> pd.DataFrame:
    filename = "repairs.csv"
    df = pd.read_csv(raw_dir / filename)
    required = [
        "repair_id", "point_id", "reported_on", "fixed_on",
        "fault_type", "cost_xaf", "technician_id", "funded_by",
    ]
    validate_columns(df, required, filename)
    return df


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

    water_points = load_water_points(config.raw_dir)
    logger.info("Loaded water_points.csv with %d rows", len(water_points))

    inspections = load_inspections(config.raw_dir)
    logger.info("Loaded inspections.csv with %d rows", len(inspections))

    repairs = load_repairs(config.raw_dir)
    logger.info("Loaded repairs.csv with %d rows", len(repairs))


if __name__ == "__main__":
    main()