import numpy as np

from models.losses import binary_cross_entropy


targets = np.array([1, 0, 1])
probabilities = np.array([0.9, 0.2, 0.8])

loss = binary_cross_entropy(probabilities, targets)

assert np.isclose(loss, 0.1838825394)

good_loss = binary_cross_entropy(
    np.array([0.9, 0.1]),
    np.array([1, 0]),
)

bad_loss = binary_cross_entropy(
    np.array([0.1, 0.9]),
    np.array([1, 0]),
)

assert good_loss < bad_loss

try:
    loss = binary_cross_entropy(
        np.array([0.9, 0.1]),
        np.array([2, 0]),
    )
    assert False, "Expected error"
except ValueError:
    pass

try:
    loss = binary_cross_entropy(
        np.array([0.9, 0.1]),
        np.array([1, 0, 1]),
    )
    assert False, "Expected error"
except ValueError:
    pass

try:
    loss = binary_cross_entropy(
        np.array([1.9, 0.1]),
        np.array([1, 0, 1]),
    )
    assert False, "Expected error"
except ValueError:
    pass

try:
    loss = binary_cross_entropy(
        np.array([]),
        np.array([1, 0, 1]),
    )
    assert False, "Expected error"
except ValueError:
    pass
