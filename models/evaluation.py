from models.linear_classifier import (
    accuracy,
    predict_classes,
    predict_probabilities,
)
from models.losses import binary_cross_entropy


def evaluate_binary_classifier(
    features,
    targets,
    weights,
    bias,
):
    probabilities = predict_probabilities(
        features,
        weights,
        bias,
    )

    predictions = predict_classes(
        features,
        weights,
        bias,
    )

    loss = binary_cross_entropy(probabilities, targets)
    model_accuracy = accuracy(predictions, targets)

    return loss, model_accuracy