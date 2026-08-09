import numpy as np

from datasets.linear_dataset import generate_linear_dataset
from models.evaluation import evaluate_binary_classifier
from models.training import train_binary_classifier
from matplotlib import pyplot as plt


features, targets = generate_linear_dataset(
    sample_count=500,
    seed=42,
)

weights, bias, loss_history = train_binary_classifier(
    features,
    targets,
    learning_rate=0.5,
    epochs=200,
)

assert weights.shape == (2,)
assert np.asarray(bias).ndim == 0
assert loss_history.shape == (200,)

assert loss_history[-1] < loss_history[0]
assert np.all(np.diff(loss_history) <= 0)
assert weights[0] > 0
assert weights[1] > 0

final_loss, final_accuracy = evaluate_binary_classifier(
    features,
    targets,
    weights,
    bias,
)

assert final_accuracy > 0.95
assert final_loss < loss_history[0]

print(weights)
print(bias)

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss")
plt.grid(True)
plt.show()