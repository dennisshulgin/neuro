import numpy as np

from models.gradients import binary_classifier_gradients
from models.linear_classifier import predict_probabilities
from models.losses import binary_cross_entropy


def train_binary_classifier(
    features,
    targets,
    learning_rate,
    epochs,
):
    features_array = np.asarray(features, dtype=float)
    targets_array = np.asarray(targets, dtype=float)

    if features_array.ndim != 2:
        raise ValueError("feateres array doesn't have 2 dim")

    if targets_array.ndim != 1:
        raise ValueError("targets array doesn't have 2 dim")

    if learning_rate <= 0:
        raise ValueError("learning rate must be greater than 0")

    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")

    if len(features_array[0]) == 0:
            raise ValueError("features count must be greater than 0")

    weights = np.zeros(features_array.shape[1])
    bias = 0.0
    loss_history = []

    for epoch in range(epochs):
        probabilities = predict_probabilities(features_array, weights, bias)
        loss = binary_cross_entropy(probabilities, targets_array)
        loss_history.append(loss)
        weights_gradient, bias_gradient = binary_classifier_gradients(features_array, probabilities, targets_array)
        weights = weights - learning_rate * weights_gradient
        bias = bias - learning_rate * bias_gradient

    return weights, bias, np.asarray(loss_history)