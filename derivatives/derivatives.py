from matplotlib import pyplot as plt


def linear(
    x,
):
    return 2 * x + 1


def linear_derivative(
    x,
):
    return 2


def quadratic(
    x,
):
    return x ** 2


def quadratic_derivative(
    x,
):
    return 2 * x


def loss(
    weight,
):
    return (weight - 3) ** 2


def loss_derivative(
    weight,
):
    return 2 * (weight - 3)


def main():
    x_values = [value / 10 for value in range(0, 60)]

    x_loss_weight = [loss(value) for value in x_values]
    x_loss_derivative_weight = [loss_derivative(value) for value in x_values]

    figure, axis = plt.subplots()

    axis.plot(
        x_values,
        x_loss_weight,
        label="loss(x)"
    )

    axis.plot(
        x_values,
        x_loss_derivative_weight,
        label="loss_derivative(x)"
    )

    axis.axvline(
        0, 
        color="black",
        linewidth=0.5
    )

    axis.axhline(
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