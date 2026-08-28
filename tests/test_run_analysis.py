import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from run_analysis import (
    merge_points_and_inspections,
    merge_repairs,
    MissingColumnError,
    validate_columns,
)


def test_merge_points_and_inspections_row_count():
    water_points = pd.DataFrame({
        "point_id": ["P1", "P2"],
        "village": ["A", "B"],
    })
    inspections = pd.DataFrame({
        "point_id": ["P1", "P1", "P2"],
        "inspected_on": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-15"]),
    })
    merged = merge_points_and_inspections(water_points, inspections)
    assert len(merged) == 3


def test_merge_repairs_row_count():
    analysis_table = pd.DataFrame({
        "point_id": ["P1", "P1", "P2"],
        "village": ["A", "A", "B"],
        "division": ["D1", "D1", "D2"],
        "point_type": ["borehole", "borehole", "spring"],
    })
    repairs = pd.DataFrame({
        "point_id": ["P1", "P2", "P2"],
        "cost_xaf": [1000, 2000, 3000],
    })
    merged = merge_repairs(analysis_table, repairs)
    assert len(merged) == 3


def test_missing_column_raises():
    df = pd.DataFrame({"point_id": ["P1"]})
    with pytest.raises(MissingColumnError):
        validate_columns(df, ["point_id", "village"], "test.csv")