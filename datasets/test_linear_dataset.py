import numpy as np

from linear_dataset import (
    create_targets,
    generate_linear_dataset,
)


features, targets = generate_linear_dataset(
    sample_count=100,
    seed=42,
)

assert features.shape == (100, 2)
assert targets.shape == (100,)

assert set(np.unique(targets)).issubset({0, 1})

assert np.array_equal(
    targets,
    create_targets(features),
)

same_features, same_targets = generate_linear_dataset(
    sample_count=100,
    seed=42,
)

assert np.array_equal(
    features,
    same_features,
)

assert np.array_equal(
    targets,
    same_targets,
)

assert np.array_equal(
    create_targets(
        np.array(
            [
                [0.7, -0.2],
                [-0.1, -0.4],
                [0.3, -0.3],
            ]
        )
    ),
    np.array([1, 0, 0]),
)

try:
    generate_linear_dataset(
        sample_count=0,
    )
    assert False, "Ожидался ValueError"
except ValueError:
    pass

try:
    create_targets(
        np.array(
            [
                [1, 2, 3],
            ]
        )
    )
    assert False, "Ожидался ValueError"
except ValueError:
    pass

print("Все проверки девятого дня пройдены")