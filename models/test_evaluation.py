import numpy as np

from datasets.linear_dataset import generate_linear_dataset
from models.evaluation import evaluate_binary_classifier


features, targets = generate_linear_dataset(
    sample_count=500,
    seed=42,
)

weights = np.array([1.0, 1.0])
bias = 0.0

confident_loss, confident_accuracy = evaluate_binary_classifier(features, targets, weights, bias)
print(confident_loss, confident_accuracy)

weights = np.array([5.0, 5.0])
bias = 0.0

confident_loss, confident_accuracy = evaluate_binary_classifier(features, targets, weights, bias)
print(confident_loss, confident_accuracy)

weights = np.array([0.0, 0.0])
bias = 0.0

confident_loss, confident_accuracy = evaluate_binary_classifier(features, targets, weights, bias)
print(confident_loss, confident_accuracy)

weights = np.array([-1.0, -1.0])
bias = 0.0

confident_loss, confident_accuracy = evaluate_binary_classifier(features, targets, weights, bias)
print(confident_loss, confident_accuracy)
