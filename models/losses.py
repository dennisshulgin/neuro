import numpy as np
import math


def binary_cross_entropy(
    probabilities,
    targets,
):
    probabilities_array = np.asarray(probabilities, dtype=float)
    targets_array = np.asarray(targets, dtype=float)

    if probabilities_array.ndim != 1:
        raise ValueError("Array must have 1 dim")

    if targets_array.ndim != 1:
        raise ValueError("Array must have 1 dim")

    if probabilities_array.shape != targets_array.shape:
        raise ValueError("Invalid shapes")

    epsilon = 1e-15  # Очень маленькое число (0.000000000000001)
    safe_probabilities = np.clip(
        probabilities_array,  # Исходные предсказания модели
        epsilon,              # Минимальное значение (0 + эпсилон)
        1 - epsilon,          # Максимальное значение (1 - эпсилон)
    )

    result = 0
    for i in range(0, len(safe_probabilities)):
        if targets_array[i] not in (0, 1):
            raise ValueError("Invalid target")
        if safe_probabilities[i] < 0 or safe_probabilities[i] > 1:
            raise ValueError("Invalid probability")

        result += -(targets_array[i] * math.log(safe_probabilities[i]) + (1 - targets_array[i]) * math.log(1 - safe_probabilities[i]))

    return result / len(safe_probabilities)


    

        
