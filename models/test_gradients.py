import numpy as np

from models.gradients import binary_classifier_gradients
from models.linear_classifier import predict_probabilities
from models.losses import binary_cross_entropy


features = np.array([
    [2.0, -1.0],
    [-1.0, -2.0],
    [1.0, 1.0],
])

targets = np.array([1.0, 0.0, 1.0])

weights = np.array([0.0, 0.0])
bias = 0.0

probabilities = predict_probabilities(
    features,
    weights,
    bias,
)

assert np.allclose(
    probabilities,
    np.array([0.5, 0.5, 0.5]),
)

weights_gradient, bias_gradient = binary_classifier_gradients(
    features,
    probabilities,
    targets,
)

loss = binary_cross_entropy(
    probabilities,
    targets
)

assert np.allclose(
    weights_gradient,
    np.array([-2 / 3, -1 / 3]),
)

assert np.isclose(
    bias_gradient,
    -1 / 6,
)

learning_rate = 0.1
new_weights = weights - learning_rate * weights_gradient
new_bias = bias - learning_rate * bias_gradient


new_probabilities = predict_probabilities(
    features,
    new_weights,
    new_bias,
)

new_loss = binary_cross_entropy(
    new_probabilities,
    targets
)

assert new_loss < loss