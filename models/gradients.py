import numpy as np


def binary_classifier_gradients(
    features,
    probabilities,
    targets,
):
    features_array = np.asarray(features, dtype=float)
    probabilities_array = np.asarray(probabilities, dtype=float)
    targets_array = np.asarray(targets, dtype=float)

    if probabilities_array.shape[0] != targets_array.shape[0]: 
        raise ValueError("Invalid array size")

    errors = probabilities_array - targets_array
    sample_count = features_array.shape[0]
    weights_gradient = features_array.T @ errors / sample_count
    bias_gradient = errors.mean()

    return (weights_gradient, bias_gradient)
