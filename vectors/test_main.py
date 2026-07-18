from main import add_vectors
from main import multiply_by_scalar
from main import dot_product
from main import neuron_output


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