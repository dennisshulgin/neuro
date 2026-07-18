from vectors import add_vectors
from vectors import multiply_by_scalar


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