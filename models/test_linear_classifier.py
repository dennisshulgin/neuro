import numpy as np

from datasets.linear_dataset import generate_linear_dataset
from models.linear_classifier import (
    accuracy,
    linear_scores,
    predict_classes,
)


features = np.array(
    [
        [0.8, -0.3],
        [-0.2, -0.5],
        [0.4, -0.4],
    ]
)

weights = np.array([1, 1])
bias = 0

assert np.allclose(
    linear_scores(
        features,
        weights,
        bias,
    ),
    np.array([0.5, -0.7, 0.0]),
)

assert np.array_equal(
    predict_classes(
        features,
        weights,
        bias,
    ),
    np.array([1, 0, 0]),
)

assert np.isclose(
    accuracy(
        np.array([1, 0, 0]),
        np.array([1, 1, 0]),
    ),
    2 / 3,
)

generated_features, generated_targets = generate_linear_dataset(
    sample_count=1000,
    seed=42,
)

generated_predictions = predict_classes(
    generated_features,
    np.array([1, 1]),
    0,
)

assert accuracy(
    generated_predictions,
    generated_targets,
) == 1.0

try:
    linear_scores(
        np.array([1, 2]),
        np.array([1, 1]),
        0,
    )
    assert False, "Ожидался ValueError"
except ValueError:
    pass

try:
    accuracy(
        np.array([1, 0]),
        np.array([1, 0, 1]),
    )
    assert False, "Ожидался ValueError"
except ValueError:
    pass

print("Все проверки десятого дня пройдены")