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


class DateParsingError(Exception):
    def __init__(self, column: str, bad_values: list):
        self.column = column
        self.bad_values = bad_values
        super().__init__(
            f"Could not parse {len(bad_values)} value(s) in column '{column}': "
            f"{bad_values[:5]}"
        )


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


def parse_flexible_date(series: pd.Series, column: str) -> pd.Series:
    known_formats = ["%Y-%m-%d", "%d %b %Y", "%d/%m/%Y"]
    parsed = pd.Series(pd.NaT, index=series.index)
    remaining = series.copy()

    for fmt in known_formats:
        mask = remaining.notna() & parsed.isna()
        attempt = pd.to_datetime(remaining[mask], format=fmt, errors="coerce")
        parsed.loc[attempt.notna().index[attempt.notna()]] = attempt[attempt.notna()]

    still_bad = series[series.notna() & parsed.isna()]
    if not still_bad.empty:
        raise DateParsingError(column, still_bad.tolist())

    return parsed


def load_water_points(raw_dir: Path) -> pd.DataFrame:
    filename = "water_points.csv"
    df = pd.read_csv(raw_dir / filename)
    required = [
        "point_id", "village", "division", "point_type", "installed_on",
        "depth_m", "households_served", "managed_by", "latitude", "longitude",
    ]
    validate_columns(df, required, filename)
    df["installed_on"] = parse_flexible_date(df["installed_on"], "installed_on")
    return df


def load_inspections(raw_dir: Path) -> pd.DataFrame:
    filename = "inspections.csv"
    df = pd.read_csv(raw_dir / filename)
    required = [
        "inspection_id", "point_id", "inspected_on", "functional",
        "water_quality", "queue_minutes", "inspector_id",
    ]
    validate_columns(df, required, filename)
    df["inspected_on"] = parse_flexible_date(df["inspected_on"], "inspected_on")
    return df


def load_repairs(raw_dir: Path) -> pd.DataFrame:
    filename = "repairs.csv"
    df = pd.read_csv(raw_dir / filename)
    required = [
        "repair_id", "point_id", "reported_on", "fixed_on",
        "fault_type", "cost_xaf", "technician_id", "funded_by",
    ]
    validate_columns(df, required, filename)
    df["reported_on"] = parse_flexible_date(df["reported_on"], "reported_on")
    df["fixed_on"] = parse_flexible_date(df["fixed_on"], "fixed_on")
    return df


def resolve_duplicate_inspections(
    inspections: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    subset = ["point_id", "inspected_on"]
    is_duplicate = inspections.duplicated(subset=subset, keep=False)
    duplicate_count = is_duplicate.sum()
    if duplicate_count > 0:
        logger.warning(
            "Found %d duplicate inspection row(s) on point_id + inspected_on",
            duplicate_count,
        )
    deduped = inspections.sort_values("inspection_id").drop_duplicates(
        subset=subset, keep="last"
    )
    return deduped


def reconcile_orphaned_inspections(
    inspections: pd.DataFrame, water_points: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    valid_ids = set(water_points["point_id"])
    is_orphan = ~inspections["point_id"].isin(valid_ids)
    orphan_count = is_orphan.sum()
    if orphan_count > 0:
        logger.warning(
            "Dropping %d inspection row(s) referencing unknown point_id values",
            orphan_count,
        )
    return inspections[~is_orphan].copy()


def merge_points_and_inspections(
    water_points: pd.DataFrame, inspections: pd.DataFrame
) -> pd.DataFrame:
    # Expected cardinality: one water point -> many inspections
    merged = water_points.merge(
        inspections, on="point_id", how="left", validate="one_to_many"
    )
    return merged


def flag_unresolved_repairs(
    repairs: pd.DataFrame, config: Config, logger: logging.Logger
) -> pd.DataFrame:
    repairs = repairs.copy()
    repairs["is_unresolved"] = repairs["fixed_on"].isna()
    unresolved_count = repairs["is_unresolved"].sum()
    logger.warning(
        "%d repair(s) have no fixed_on date and are treated as still unresolved",
        unresolved_count,
    )

    cutoff = pd.Timestamp(config.date_cutoff)
    effective_end = repairs["fixed_on"].fillna(cutoff)
    repairs["downtime_days"] = (effective_end - repairs["reported_on"]).dt.days
    return repairs


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

    inspections = resolve_duplicate_inspections(inspections, logger)
    logger.info("Inspections after removing duplicates: %d rows", len(inspections))

    inspections = reconcile_orphaned_inspections(inspections, water_points, logger)
    logger.info("Inspections after removing orphans: %d rows", len(inspections))

    merged = merge_points_and_inspections(water_points, inspections)
    logger.info("Merged water_points + inspections: %d rows", len(merged))

    repairs = flag_unresolved_repairs(repairs, config, logger)
    logger.info(
        "Repairs processed: %d total, %d unresolved",
        len(repairs), repairs["is_unresolved"].sum(),
    )


if __name__ == "__main__":
    main()
