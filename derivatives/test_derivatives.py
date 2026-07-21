from derivatives import (
    linear_derivative,
    loss,
    loss_derivative,
    quadratic_derivative,
)


assert linear_derivative(-100) == 2
assert linear_derivative(0) == 2
assert linear_derivative(100) == 2

assert quadratic_derivative(-2) == -4
assert quadratic_derivative(0) == 0
assert quadratic_derivative(2) == 4

assert loss(0) == 9
assert loss(3) == 0
assert loss(5) == 4

assert loss_derivative(0) == -6
assert loss_derivative(3) == 0
assert loss_derivative(5) == 4

print("Все проверки шестого дня пройдены")