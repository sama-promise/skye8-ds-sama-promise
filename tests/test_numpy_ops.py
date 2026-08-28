import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from numpy_ops import (
    min_max_scale,
    z_score_standardize,
    queue_time_deviation,
    pairwise_distance,
    nearest_neighbors,
)


def test_min_max_matches_sklearn():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 100, size=30)
    mine = min_max_scale(x)
    sklearn_result = MinMaxScaler().fit_transform(x.reshape(-1, 1)).ravel()
    assert np.allclose(mine, sklearn_result, atol=1e-9)


def test_min_max_output_range():
    x = np.array([5.0, 10.0, 15.0, 20.0])
    result = min_max_scale(x)
    assert result.min() == 0.0
    assert result.max() == 1.0


def test_zscore_matches_sklearn():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 100, size=30)
    mine = z_score_standardize(x)
    sklearn_result = StandardScaler().fit_transform(x.reshape(-1, 1)).ravel()
    assert np.allclose(mine, sklearn_result, atol=1e-9)


def test_zscore_mean_near_zero():
    x = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    result = z_score_standardize(x)
    assert abs(result.mean()) < 1e-9


def test_queue_deviation_sums_to_zero_per_group():
    point_ids = np.array(["A", "A", "A", "B", "B"])
    queue = np.array([10.0, 20.0, 30.0, 5.0, 15.0])
    result = queue_time_deviation(point_ids, queue)
    assert np.isclose(result[:3].sum(), 0)
    assert np.isclose(result[3:].sum(), 0)


def test_pairwise_distance_diagonal_is_zero():
    coords = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
    result = pairwise_distance(coords)
    assert np.allclose(np.diag(result), 0.0)


def test_pairwise_distance_known_value():
    coords = np.array([[0.0, 0.0], [3.0, 4.0]])
    result = pairwise_distance(coords)
    assert np.isclose(result[0, 1], 5.0)


def test_nearest_neighbors_excludes_self():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
    neighbors = nearest_neighbors(coords, k=2)
    for i, row in enumerate(neighbors):
        assert i not in row