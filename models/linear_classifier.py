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
        raise

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
    matches = predictions == targets
    return matches.mean()