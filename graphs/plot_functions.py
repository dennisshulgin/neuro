from math import exp
from matplotlib import pyplot as plt


def linear(x):
    return 2 * x + 1


def quadratic(x):
    return x ** 2


def exponential(x):
    return exp(x)


def main():
    x_values = [value / 10 for value in range(-20, 21)]

    linear_values = [linear(value) for value in x_values]
    quadratic_values = [quadratic(value) for value in x_values]
    exponential_values = [exponential(value) for value in x_values]

    figure, axis = plt.subplots()

    axis.plot(
        x_values,
        linear_values,
        label="2x + 1"
    )

    axis.plot(
        x_values,
        quadratic_values,
        label="x^2"
    )

    axis.plot(
        x_values,
        exponential_values,
        label="exp(x)"
    )

    axis.axhline(
        0, 
        color="black",
        linewidth=0.5
    )

    axis.axvline(
        0, 
        color="black",
        linewidth=0.5
    )

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Functions")
    axis.grid(True)
    axis.legend()

    plt.show()


if __name__ == '__main__':
    main()

