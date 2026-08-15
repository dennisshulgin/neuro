from datasets.linear_dataset import generate_linear_dataset
from datasets.split import train_test_split
from models.training import train_binary_classifier
from models.evaluation import evaluate_binary_classifier




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

train_loss, train_accuracy = evaluate_binary_classifier(
    train_features,
    train_targets,
    weights,
    bias,
)

test_loss, test_accuracy = evaluate_binary_classifier(
    test_features,
    test_targets,
    weights,
    bias,
)

print(train_loss)
print(test_loss)