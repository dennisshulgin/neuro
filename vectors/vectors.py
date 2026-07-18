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