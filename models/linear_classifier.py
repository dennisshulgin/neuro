import numpy as np


def linear_scores(
    features,
    weights,
    bias,
):
    """
    Возвращает score для каждого объекта.
    """
    if features.ndim != 2:
        raise ValueError("Features must have 2 dim")

    if weights.ndim != 1:
        raise ValueError("Weights must have 1 dim")

    if features.shape[1] != weights.shape[0]:
        raise ValueError("Invalid arrays")

    return features @ weights + bias


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

    matches = predictions == targets
    return matches.mean()