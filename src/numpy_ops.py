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


def nearest_neighbors(coords: np.ndarray, k: int = 3) -> np.ndarray:
    distances = pairwise_distance(coords)
    np.fill_diagonal(distances, np.inf)
    neighbor_indices = np.argsort(distances, axis=1)[:, :k]
    return neighbor_indices


if __name__ == "__main__":
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    rng = np.random.default_rng(42)
    sample = rng.uniform(0, 100, size=50)

    my_minmax = min_max_scale(sample)
    sk_minmax = MinMaxScaler().fit_transform(sample.reshape(-1, 1)).ravel()
    minmax_matches = np.allclose(my_minmax, sk_minmax, atol=1e-9)
    print("Min-max matches scikit-learn:", minmax_matches)

    my_zscore = z_score_standardize(sample)
    sk_zscore = StandardScaler().fit_transform(sample.reshape(-1, 1)).ravel()
    zscore_matches = np.allclose(my_zscore, sk_zscore, atol=1e-9)
    print("Z-score matches scikit-learn:", zscore_matches)

    test_points = np.array(["A", "A", "A", "B", "B"])
    test_queue = np.array([10.0, 20.0, 30.0, 5.0, 15.0])
    deviations = queue_time_deviation(test_points, test_queue)
    print("Queue time deviations:", deviations)
    print("Deviations sum to ~0 per group:", np.allclose(deviations[:3].sum(), 0))

    test_coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [10.0, 10.0],
    ])
    dist_matrix = pairwise_distance(test_coords)
    print("Pairwise distance matrix:\n", dist_matrix)

    neighbors = nearest_neighbors(test_coords, k=2)
    print("2 nearest neighbors for each point:\n", neighbors)