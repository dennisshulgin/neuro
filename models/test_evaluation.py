import numpy as np

from datasets.linear_dataset import generate_linear_dataset
from models.evaluation import evaluate_binary_classifier


features, targets = generate_linear_dataset(
    sample_count=500,
    seed=42,
)

weights = np.array([1.0, 1.0])
bias = 0.0

correct_loss, correct_accuracy = evaluate_binary_classifier(features, targets, weights, bias)

weights = np.array([5.0, 5.0])
bias = 0.0

confident_loss, confident_accuracy = evaluate_binary_classifier(features, targets, weights, bias)

weights = np.array([0.0, 0.0])
bias = 0.0

zero_loss, zero_accuracy = evaluate_binary_classifier(features, targets, weights, bias)

weights = np.array([-1.0, -1.0])
bias = 0.0

wrong_loss, wrong_accuracy = evaluate_binary_classifier(features, targets, weights, bias)

assert np.isclose(correct_accuracy, 1.0)
assert np.isclose(confident_accuracy, correct_accuracy)
assert np.isclose(wrong_accuracy, 0.0)

assert wrong_accuracy < zero_accuracy
assert zero_accuracy < correct_accuracy

assert confident_loss < correct_loss
assert correct_loss < zero_loss
assert zero_loss < wrong_loss