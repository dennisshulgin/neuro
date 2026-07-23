import numpy as np


def add_vectors(
    first,
    second,
):
    """
    Складывает два вектора с помощью NumPy.

    При разных shape выбрасывает ValueError.
    """
    a = np.array(first)
    b = np.array(second)

    if a.shape != b.shape:
        raise ValueError("Arrays are not equal")

    return a + b


def dot_product(
    first,
    second,
):
    """
    Возвращает скалярное произведение.

    При разных shape выбрасывает ValueError.
    """
    a = np.array(first)
    b = np.array(second)
    
    if a.shape != b.shape:
        raise ValueError("Arrays are not equal")
    
    return a @ b
    


def dense_layer_output(
    features,
    weights,
    biases,
):
    """
    Вычисляет weights @ features + biases.
    """
    a = np.array(weights)
    b = np.array(features)
    c = np.array(biases)

    return a @ b + c