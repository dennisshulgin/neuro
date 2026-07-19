from main import (
    add_vectors,
    dense_layer_output,
    dot_product,
    matrix_vector_product,
    mean,
    multiply_by_scalar,
    neuron_output,
    variance,
    standard_deviation,
)

assert add_vectors([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
assert add_vectors([-1, 5], [1, 10]) == [0, 15]

assert multiply_by_scalar([1, 2, 3], 2) == [2, 4, 6]
assert multiply_by_scalar([-2, 4], 0.5) == [-1, 2]

try:
    add_vectors([1, 2], [3, 4, 5])
    assert False, "Ожидался ValueError"
except ValueError:
    pass

print("Все проверки пройдены")

assert dot_product([1, 2, 3], [4, 5, 6]) == 32
assert dot_product([2, -1], [3, 4]) == 2
assert dot_product([0, 0], [100, -100]) == 0

assert neuron_output([2, 3], [0.5, -1], 2) == 0
assert neuron_output([1, 0, 1], [2, 10, -3], 0.5) == -0.5

try:
    dot_product([1, 2], [3])
    assert False, "Ожидался ValueError"
except ValueError:
    pass

print("Все проверки второго дня пройдены")

assert matrix_vector_product(
    [
        [1, 2, 3],
        [4, 5, 6],
    ],
    [10, 20, 30],
) == [140, 320]

assert matrix_vector_product(
    [
        [1, 0],
        [0, 1],
    ],
    [7, 9],
) == [7, 9]

assert dense_layer_output(
    [2, 4],
    [
        [1.0, 2.0],
        [-1.0, 3.0],
        [0.5, 0.5],
    ],
    [0, 1, -1],
) == [10, 11, 2]

try:
    matrix_vector_product(
        [[1, 2, 3]],
        [1, 2],
    )
    assert False, "Ожидался ValueError"
except ValueError:
    pass

try:
    dense_layer_output(
        [1, 2],
        [
            [1, 0],
            [0, 1],
        ],
        [0],
    )
    assert False, "Ожидался ValueError"
except ValueError:
    pass

print("Все проверки третьего дня пройдены")

assert mean([2, 4, 6]) == 4
assert mean([1, 2, 3, 4]) == 2.5
assert mean([-10, 10]) == 0

assert variance([1, 2, 3, 4]) == 1.25
assert variance([5, 5, 5]) == 0

assert abs(
    standard_deviation([1, 2, 3, 4]) - 1.118033988749895
) < 0.000000001

assert standard_deviation([5, 5, 5]) == 0

try:
    mean([])
    assert False, "Ожидался ValueError"
except ValueError:
    pass

try:
    variance([])
    assert False, "Ожидался ValueError"
except ValueError:
    pass

print("Все проверки четвёртого дня пройдены")