from datasets.linear_dataset import generate_linear_dataset
from datasets.split import train_test_split
from models.training import train_binary_classifier
from models.evaluation import evaluate_binary_classifier
from datasets.noise import add_label_noise

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

noisy_train_targets = add_label_noise(
    train_targets,
    noise_ratio=0.1,
    seed=42,
)

weights, bias, loss_history = train_binary_classifier(
    train_features,
    noisy_train_targets,
    learning_rate=0.5,
    epochs=200,
)

test_loss, test_accuracy = evaluate_binary_classifier(
    test_features,
    test_targets,
    weights,
    bias,
)

assert test_accuracy > 0.85
assert test_accuracy < 1.0