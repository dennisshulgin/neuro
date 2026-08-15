import numpy as np

from datasets.linear_dataset import generate_linear_dataset
from datasets.split import train_test_split
from models.linear_classifier import predict_probabilities
from models.training import train_binary_classifier

features, targets = generate_linear_dataset(
    sample_count=500,
    seed=42,
)

train_features, test_features, train_targets, test_targets = train_test_split(
    features,
    targets,
    test_ratio=0.2,
    seed=42,
)

weights, bias, loss_history = train_binary_classifier(
    train_features,
    train_targets,
    learning_rate=0.5,
    epochs=200,
)

point = np.array([[1.0, 2.0]])

probability = predict_probabilities(
    point,
    weights,
    bias,
)[0]

predicted_class = 1 if probability >= 0.5 else 0

print("probability:", probability)
print("predicted_class:", predicted_class)