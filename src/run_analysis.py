import argparse
import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt

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


def clean_depth_m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned = (
        df["depth_m"]
        .astype(str)
        .str.replace("m", "", regex=False)
        .str.strip()
    )
    df["depth_m"] = pd.to_numeric(cleaned, errors="coerce")
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


def normalize_functional_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    true_values = {"yes", "true", "1", "1.0"}
    df["functional"] = (
        df["functional"].astype(str).str.strip().str.lower().isin(true_values)
    )
    return df


def clean_queue_minutes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned = (
        df["queue_minutes"]
        .astype(str)
        .str.replace("min", "", regex=False)
        .str.strip()
    )
    df["queue_minutes"] = pd.to_numeric(cleaned, errors="coerce")
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


def clean_cost_xaf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned = (
        df["cost_xaf"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["cost_xaf"] = pd.to_numeric(cleaned, errors="coerce")
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


def merge_repairs(analysis_table: pd.DataFrame, repairs: pd.DataFrame) -> pd.DataFrame:
    # Expected cardinality: one water point -> many repairs
    points_only = analysis_table[
        ["point_id", "village", "division", "point_type"]
    ].drop_duplicates(subset="point_id")
    merged = points_only.merge(
        repairs, on="point_id", how="left", validate="one_to_many"
    )
    return merged


def build_monthly_inspection_summary(inspections_merged: pd.DataFrame) -> pd.DataFrame:
    indexed = inspections_merged.set_index("inspected_on")
    summary = indexed.resample("ME").agg(
        functionality_rate=("functional", "mean"),
        mean_queue_minutes=("queue_minutes", "mean"),
    )
    return summary


def build_monthly_repair_summary(repairs: pd.DataFrame) -> pd.DataFrame:
    indexed = repairs.set_index("reported_on")
    summary = indexed.resample("ME").agg(
        repairs_reported=("repair_id", "count"),
        repairs_completed=("is_unresolved", lambda s: (~s).sum()),
        total_repair_cost=("cost_xaf", "sum"),
    )
    return summary


def build_monthly_summary(
    inspections_merged: pd.DataFrame, repairs: pd.DataFrame
) -> pd.DataFrame:
    inspection_part = build_monthly_inspection_summary(inspections_merged)
    repair_part = build_monthly_repair_summary(repairs)
    combined = inspection_part.join(repair_part, how="outer")
    return combined

def plot_functionality_by_month(monthly_summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_summary.index, monthly_summary["functionality_rate"], marker="o")
    ax.set_title("Figure 1: Functionality rate by month (2024-2026)\nQuestion: Is there a seasonal pattern in point functionality?")
    ax.set_xlabel("Month")
    ax.set_ylabel("Functionality rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig1_functionality_by_month.png", dpi=150)
    plt.close(fig)


def add_village_rank(inspections_merged: pd.DataFrame) -> pd.DataFrame:
    df = inspections_merged.copy()
    df["village_functionality_rank"] = (
        df.groupby("village")["functional"]
        .transform(lambda s: s.rank(ascending=False, method="min"))
    )
    return df


def add_division_cost_share(repairs_merged: pd.DataFrame) -> pd.DataFrame:
    df = repairs_merged.copy()
    division_totals = df.groupby("division")["cost_xaf"].transform("sum")
    df["division_cost_share"] = df["cost_xaf"] / division_totals
    return df


def reduce_memory_footprint(merged: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    df = merged.copy()

    before_bytes = df.memory_usage(deep=True).sum()

    for col in ["village", "division", "point_type", "managed_by", "water_quality"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in ["depth_m", "households_served", "latitude", "longitude", "queue_minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

    after_bytes = df.memory_usage(deep=True).sum()
    reduction_pct = 100 * (1 - after_bytes / before_bytes)
    logger.info(
        "Memory usage: %d bytes -> %d bytes (%.1f%% reduction)",
        before_bytes, after_bytes, reduction_pct,
    )
    return df


def pivot_melt_fault_types(repairs: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    repairs = repairs.copy()
    repairs["report_month"] = repairs["reported_on"].dt.to_period("M").astype(str)

    pivoted = repairs.pivot_table(
        index="report_month", columns="fault_type",
        values="repair_id", aggfunc="count", fill_value=0,
    )

    melted = pivoted.reset_index().melt(
        id_vars="report_month", var_name="fault_type", value_name="count"
    )

    pivot_total = pivoted.to_numpy().sum()
    melt_total = melted["count"].sum()
    logger.info(
        "Pivot/melt round trip totals match: pivot=%d, melted=%d",
        pivot_total, melt_total,
    )
    assert pivot_total == melt_total, "Pivot/melt round trip lost data!"

    return melted


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

    water_points = clean_depth_m(water_points)
    logger.info("Cleaned depth_m column to numeric")

    inspections = load_inspections(config.raw_dir)
    logger.info("Loaded inspections.csv with %d rows", len(inspections))

    inspections = normalize_functional_column(inspections)
    logger.info("Normalized functional column to boolean")

    inspections = clean_queue_minutes(inspections)
    logger.info("Cleaned queue_minutes column to numeric")

    repairs = load_repairs(config.raw_dir)
    logger.info("Loaded repairs.csv with %d rows", len(repairs))

    repairs = clean_cost_xaf(repairs)
    logger.info("Cleaned cost_xaf column to numeric")

    inspections = resolve_duplicate_inspections(inspections, logger)
    logger.info("Inspections after removing duplicates: %d rows", len(inspections))

    inspections = reconcile_orphaned_inspections(inspections, water_points, logger)
    logger.info("Inspections after removing orphans: %d rows", len(inspections))

    merged = merge_points_and_inspections(water_points, inspections)
    logger.info("Merged water_points + inspections: %d rows", len(merged))

    merged = add_village_rank(merged)
    logger.info("Added village_functionality_rank via transform (row count unchanged: %d)", len(merged))

    merged = reduce_memory_footprint(merged, logger)

    repairs = flag_unresolved_repairs(repairs, config, logger)
    logger.info(
        "Repairs processed: %d total, %d unresolved",
        len(repairs), repairs["is_unresolved"].sum(),
    )

    repairs_merged = merge_repairs(merged, repairs)
    logger.info("Merged water_points + repairs: %d rows", len(repairs_merged))

    repairs_merged = add_division_cost_share(repairs_merged)
    logger.info("Added division_cost_share via transform (row count unchanged: %d)", len(repairs_merged))

    fault_type_long = pivot_melt_fault_types(repairs, logger)
    logger.info("Pivot/melt round trip produced %d long-form rows", len(fault_type_long))

    monthly_summary = build_monthly_summary(merged, repairs)
    logger.info("Monthly summary built with %d rows (months)", len(monthly_summary))
    logger.info("Monthly summary preview:\n%s", monthly_summary.head())

    plot_functionality_by_month(monthly_summary, config.figures_dir)
    logger.info("Saved fig1_functionality_by_month.png")


if __name__ == "__main__":
    main()