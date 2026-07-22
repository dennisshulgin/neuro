from neuron import (
    squared_error,
    squared_error_derivative,
    evaluate_neuron
)

assert squared_error(
    3,
    5,
) == 4

assert squared_error_derivative(
    3,
    5,
) == -4

assert squared_error_derivative(
    7,
    5,
) == 4

assert squared_error_derivative(
    5,
    5,
) == 0

assert evaluate_neuron(
    [2, -1],
    [3, 4],
    1,
    5,
) == (3, 4, -4)

assert evaluate_neuron(
    [2, -1],
    [3, 4],
    1,
    3,
) == (3, 0, 0)

print("Контрольная точка первой недели пройдена")