def add_vectors(a, b):
    if len(a) != len(b):
        raise ValueError("Lengths of arrays are not equal")
    
    result = []

    for i in range(len(a)):
        result.append(a[i] + b[i])
    
    return result


def multiply_by_scalar(vector, scalar):
    result = []
    for i in range(len(vector)):
        result.append(vector[i] * scalar)
    return result

def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Lengths of arrays are not equal")
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

def neuron_output(features, weights, bias):
    return dot_product(features, weights) + bias

def matrix_vector_product(matrix, vector):
    result = []
    for row in matrix:
        result.append(dot_product(row, vector))
    return result

def dense_layer_output(features, weights, biases):
    return add_vectors(matrix_vector_product(weights, features), biases)
