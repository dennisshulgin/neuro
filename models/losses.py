import numpy as np


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

    if np.any((probabilities_array < 0) | (probabilities_array > 1)):
        raise ValueError("Invalid values")

    if np.any((targets_array != 1) & (targets_array != 0)):
            raise ValueError("Invalid values")

    if len(probabilities_array) == 0 | len(targets_array) == 0:
         raise ValueError("Array is empty")
    
    epsilon = 1e-15  # Очень маленькое число (0.000000000000001)
    safe_probabilities = np.clip(
        probabilities_array,  # Исходные предсказания модели
        epsilon,              # Минимальное значение (0 + эпсилон)
        1 - epsilon,          # Максимальное значение (1 - эпсилон)
    )

    losses = -(targets_array * np.log(safe_probabilities) + (1 - targets_array) * np.log(1 - safe_probabilities))
    return losses.mean()


    

        
