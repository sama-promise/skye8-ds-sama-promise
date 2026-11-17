import timeit

import numpy as np
import pandas as pd


def min_max_scale(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    return (x - x.min()) / (x.max() - x.min())


def z_score_standardize(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    return (x - x.mean()) / x.std(ddof=0)


def queue_time_deviation(point_ids: np.ndarray, queue_minutes: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({"point_id": point_ids, "queue_minutes": queue_minutes})
    point_means = df.groupby("point_id")["queue_minutes"].transform("mean")
    deviation = df["queue_minutes"].to_numpy() - point_means.to_numpy()
    return deviation


def pairwise_distance(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=-1))
    return distances


def pairwise_distance_loop(coords: np.ndarray) -> np.ndarray:
    n = len(coords)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(((coords[i] - coords[j]) ** 2).sum())
    return distances


def nearest_neighbors(coords: np.ndarray, k: int = 3) -> np.ndarray:
    distances = pairwise_distance(coords)
    np.fill_diagonal(distances, np.inf)
    neighbor_indices = np.argsort(distances, axis=1)[:, :k]
    return neighbor_indices


def benchmark_distance_methods(coords: np.ndarray, number: int = 3) -> tuple[float, float]:
    loop_time = timeit.timeit(lambda: pairwise_distance_loop(coords), number=number)
    vector_time = timeit.timeit(lambda: pairwise_distance(coords), number=number)
    return loop_time, vector_time


if __name__ == "__main__":
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    rng = np.random.default_rng(42)
    sample = rng.uniform(0, 100, size=50)

    my_minmax = min_max_scale(sample)
    sk_minmax = MinMaxScaler().fit_transform(sample.reshape(-1, 1)).ravel()
    print("Min-max matches scikit-learn:", np.allclose(my_minmax, sk_minmax, atol=1e-9))

    my_zscore = z_score_standardize(sample)
    sk_zscore = StandardScaler().fit_transform(sample.reshape(-1, 1)).ravel()
    print("Z-score matches scikit-learn:", np.allclose(my_zscore, sk_zscore, atol=1e-9))

    test_points = np.array(["A", "A", "A", "B", "B"])
    test_queue = np.array([10.0, 20.0, 30.0, 5.0, 15.0])
    deviations = queue_time_deviation(test_points, test_queue)
    print("Queue time deviations:", deviations)

    test_coords = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0],
    ])
    print("Pairwise distance matrix:\n", pairwise_distance(test_coords))
    print("2 nearest neighbors for each point:\n", nearest_neighbors(test_coords, k=2))

    print("\nBenchmarking vectorized vs loop-based distance on 173 random points...")
    real_size_coords = rng.uniform(0, 100, size=(173, 2))
    loop_result = pairwise_distance_loop(real_size_coords)
    vector_result = pairwise_distance(real_size_coords)
    print("Loop and vectorized results match:", np.allclose(loop_result, vector_result))

    loop_time, vector_time = benchmark_distance_methods(real_size_coords, number=3)
    speedup = loop_time / vector_time
    print(f"Loop-based time: {loop_time:.4f}s")
    print(f"Vectorized time: {vector_time:.4f}s")
    print(f"Speed-up: {speedup:.1f}x")

    print("\nApplying nearest neighbors to real water point coordinates...")
    water_points_df = pd.read_csv("data/raw/water_points.csv")
    real_coords = water_points_df[["latitude", "longitude"]].to_numpy()
    real_neighbors = nearest_neighbors(real_coords, k=3)

    print(f"Computed 3 nearest neighbors for {len(real_coords)} real water points.")
    print("Example - point 0's 3 nearest neighbors (by row index):", real_neighbors[0])
    print("Point 0 coordinates:", real_coords[0])
    print("Its nearest neighbor coordinates:", real_coords[real_neighbors[0][0]])