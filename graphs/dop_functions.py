from math import exp
from matplotlib import pyplot as plt


def loss(x):
    return (x - 3) ** 2


def main():
    x_values = [value / 10 for value in range(0, 61)]
    loss_values = [loss(value) for value in x_values]

    figure, axis = plt.subplots()

    axis.plot(
        x_values,
        loss_values,
        label="loss(x)"
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
