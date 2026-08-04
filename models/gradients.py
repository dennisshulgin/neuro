import numpy as np


def binary_classifier_gradients(
    features,
    probabilities,
    targets,
):
    features_array = np.asarray(features, dtype=float)
    probabilities_array = np.asarray(probabilities, dtype=float)
    targets_array = np.asarray(targets, dtype=float)

    if features_array.ndim != 2:
        raise ValueError("features must be two-dimensional")

    if probabilities_array.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")

    if targets_array.ndim != 1:
        raise ValueError("targets must be one-dimensional")

    if probabilities_array.shape != targets_array.shape:
        raise ValueError("probabilities and targets must have equal shapes")

    if features_array.shape[0] != targets_array.shape[0]:
        raise ValueError("features and targets must contain equal object counts")

    if features_array.shape[0] == 0:
        raise ValueError("arrays must not be empty")

    errors = probabilities_array - targets_array
    sample_count = features_array.shape[0]
    weights_gradient = features_array.T @ errors / sample_count
    bias_gradient = errors.mean()

    return weights_gradient, bias_gradient
