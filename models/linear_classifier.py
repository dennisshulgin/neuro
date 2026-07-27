import numpy as np

from models.activations import sigmoid


def linear_scores(
    features,
    weights,
    bias,
):
    """
    Возвращает score для каждого объекта.
    """

    features_array = np.asarray(features)
    weights_array = np.asarray(weights)

    if features_array.ndim != 2:
        raise ValueError("Features must have 2 dim")

    if weights_array.ndim != 1:
        raise ValueError("Weights must have 1 dim")

    if features_array.shape[1] != weights_array.shape[0]:
        raise ValueError("Invalid arrays")

    if np.asarray(bias).ndim != 0:
        raise ValueError("bias must be a scalar")

    return features_array @ weights_array + bias


def predict_classes(
    features,
    weights,
    bias,
):
    """
    Возвращает 1 для score > 0, иначе 0.
    """
    scores = linear_scores(features, weights, bias)
    return (scores > 0).astype(int)


def accuracy(
    predictions,
    targets,
):
    """
    Возвращает долю правильных предсказаний.
    """
    predictions_array = np.asarray(predictions)
    targets_array = np.asarray(targets)

    if predictions_array.ndim != 1:
        raise ValueError("predictions must be one-dimensional")

    if targets_array.ndim != 1:
        raise ValueError("targets must be one-dimensional")

    if predictions_array.shape != targets_array.shape:
        raise ValueError("predictions and targets must have equal shapes")

    matches = predictions_array == targets_array
    return matches.mean()

def predict_probabilities(
    features,
    weights,
    bias,
):
    scores = linear_scores(features, weights, bias)
    return sigmoid(scores)