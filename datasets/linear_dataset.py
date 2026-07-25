import numpy as np


def create_targets(
    features,
):
    """
    Для каждой точки возвращает 1, если x1 + x2 > 0,
    иначе возвращает 0.
    """

    if features.shape[1] != 2:
        raise ValueError("Invalid columns count")

    first_coordinates = features[:, 0]
    second_coordinates = features[:, 1]

    sums = first_coordinates + second_coordinates

    return (sums > 0).astype(int)


def generate_linear_dataset(
    sample_count,
    seed=42,
):
    """
    Генерирует sample_count точек с двумя признаками
    и соответствующие targets.
    """
    if sample_count <= 0:
        raise ValueError("Sample count must be greater than 0")

    random_generator = np.random.default_rng(seed)

    features = random_generator.uniform(
        low=-1,
        high=1,
        size=(sample_count, 2),
    )

    return features, create_targets(features)