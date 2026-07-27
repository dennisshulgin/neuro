import numpy as np


def sigmoid(values):
    values_array = np.asarray(values, dtype=float)
    return 1 / (1 + np.exp(-values_array))