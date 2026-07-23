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

    if a.ndim != 2:
        raise ValueError("weights must be two-dimensional")

    if b.ndim != 1:
        raise ValueError("features must be one-dimensional")

    if c.ndim != 1:
        raise ValueError("biases must be one-dimensional")

    if a.shape[1] != b.shape[0]:
        raise ValueError("weights and features have incompatible shapes")

    if a.shape[0] != c.shape[0]:
        raise ValueError("weights and biases have incompatible shapes")

    return a @ b + c