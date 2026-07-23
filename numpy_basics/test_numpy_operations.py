import numpy as np

from numpy_operations import (
    add_vectors,
    dense_layer_output,
    dot_product,
)


assert np.array_equal(
    add_vectors(
        [1, 2, 3],
        [4, 5, 6],
    ),
    np.array([5, 7, 9]),
)

assert dot_product(
    [1, 2, 3],
    [4, 5, 6],
) == 32

assert np.allclose(
    dense_layer_output(
        [2, 4],
        [
            [1.0, 2.0],
            [-1.0, 3.0],
            [0.5, 0.5],
        ],
        [0, 1, -1],
    ),
    np.array([10, 11, 2]),
)

try:
    add_vectors(
        [1, 2],
        [1, 2, 3],
    )
    assert False, "Ожидался ValueError"
except ValueError:
    pass

try:
    dense_layer_output(
        [1, 2, 3],
        [
            [1, 0],
            [0, 1],
        ],
        [0, 0],
    )
    assert False, "Ожидался ValueError"
except ValueError:
    pass

print("Все проверки восьмого дня пройдены")