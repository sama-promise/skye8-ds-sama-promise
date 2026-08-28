import numpy as np


def min_max_scale(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    return (x - x.min()) / (x.max() - x.min())


def z_score_standardize(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    return (x - x.mean()) / x.std(ddof=0)


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