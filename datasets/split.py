import numpy as np

def split_by_ratio(arr, test_ratio):
    n = len(arr)
    test_size = int(n * test_ratio)
    train_size = n - test_size
    return arr[:train_size], arr[train_size:]

def train_test_split(
    features,
    targets,
    test_ratio=0.2,
    seed=None,
):
    features_array = np.asarray(features)
    targets_array = np.asarray(targets)

    if features_array.ndim != 2:
        raise ValueError("features must have 2 dim")

    if targets_array.ndim != 1:
        raise ValueError("targets must have 1 dim")

    if features_array.shape[0] != targets_array.shape[0]:
        raise ValueError("shape is not equal")

    if features_array.shape[0] < 2:
        raise ValueError("min two objects")

    if test_ratio > 1 or test_ratio < 0:
        raise ValueError("test ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(features_array.shape[0])
    train, test = split_by_ratio(indices, test_ratio)

    return features_array[train], features_array[test], targets_array[train], targets_array[test]