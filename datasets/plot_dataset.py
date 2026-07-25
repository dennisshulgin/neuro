from matplotlib import pyplot as plt

from linear_dataset import generate_linear_dataset


def main():
    features, targets = generate_linear_dataset(
        sample_count=200,
        seed=42,
    )

    figure, axis = plt.subplots()

    axis.scatter(
        features[:, 0],
        features[:, 1],
        c=targets,
        cmap="coolwarm",
    )

    axis.plot(
        [-1, 1],
        [1, -1],
        color="black",
        label="x₂ = -x₁",
    )

    axis.set_xlabel("x₁")
    axis.set_ylabel("x₂")
    axis.set_title("Linear classification dataset")
    axis.grid(True)
    axis.legend()

    plt.show()


if __name__ == "__main__":
    main()