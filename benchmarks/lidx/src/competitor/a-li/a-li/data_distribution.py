import numpy as np
import struct

import config


def load_keys(path: str, count: int) -> np.ndarray:
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        actual_count = min(count, size)
        keys = np.frombuffer(f.read(actual_count * 8), dtype=np.uint64)
    return keys[:actual_count].copy()


def extract_features(keys: np.ndarray) -> np.ndarray:
    keys_sorted = np.sort(keys).astype(np.float64)
    n = len(keys_sorted)

    key_min = keys_sorted[0]
    key_max = keys_sorted[-1]
    key_range = key_max - key_min
    if key_range == 0:
        key_range = 1.0

    keys_norm = (keys_sorted - key_min) / key_range

    feat_mean = np.mean(keys_norm)
    feat_std = np.std(keys_norm)
    feat_skewness = _sigmoid(_skewness(keys_norm))
    feat_kurtosis = _sigmoid(_kurtosis(keys_norm) / 10.0)
    feat_min = 0.0
    feat_max = 1.0

    stats = [feat_mean, feat_std, feat_skewness, feat_kurtosis, feat_min, feat_max]

    percentiles = np.percentile(keys_norm, [10, 25, 50, 75, 90])
    quantiles = percentiles.tolist()

    sample_points = np.linspace(key_min, key_max, config.CDF_SAMPLE_POINTS + 2)[1:-1]
    cdf_values = np.searchsorted(keys_sorted, sample_points) / n
    cdf_features = cdf_values.tolist()

    sample_size = min(1_000_000, n)
    if n > sample_size:
        indices = np.sort(np.random.choice(n, sample_size, replace=False))
        sampled = keys_sorted[indices]
    else:
        sampled = keys_sorted

    gaps = np.diff(sampled).astype(np.float64)
    if len(gaps) == 0:
        gap_features = [0.0, 0.0, 0.0, 0.0]
    else:
        gap_max = np.max(gaps)
        if gap_max == 0:
            gap_max = 1.0
        gap_mean = np.mean(gaps) / gap_max
        gap_std = np.std(gaps) / gap_max
        gap_max_norm = 1.0
        gap_mean_raw = np.mean(gaps)
        gap_cv = np.std(gaps) / gap_mean_raw if gap_mean_raw > 0 else 0.0
        gap_cv = _sigmoid(gap_cv)
        gap_features = [gap_mean, gap_std, gap_max_norm, gap_cv]

    features = np.array(stats + quantiles + cdf_features + gap_features, dtype=np.float32)
    features = np.clip(features, 0.0, 1.0)
    return features


def _skewness(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return np.mean(((x - m) / s) ** 3)


def _kurtosis(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return np.mean(((x - m) / s) ** 4) - 3.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def test_distribution():
    import os

    for dataset_path in config.DATASETS:
        if not os.path.exists(dataset_path):
            continue


        keys = load_keys(dataset_path, config.INIT_COUNT)
 
        features = extract_features(keys)
    
        labels = [
            "mean", "std", "skewness", "kurtosis", "min", "max",
            "P10", "P25", "P50", "P75", "P90",
        ] + [f"CDF_{i}" for i in range(config.CDF_SAMPLE_POINTS)] + [
            "gap_mean", "gap_std", "gap_max", "gap_cv"
        ]

        for label, val in zip(labels, features):
            print(f"  {label:>12s}: {val:.4f}")


if __name__ == "__main__":
    test_distribution()
