import numpy as np

python_list = [1, 2, 3]
np_list = np.array([1, 2, 3])

print(python_list * 2)
print(np_list * 2)

matrix = np.array(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)

print(matrix.shape)
print(matrix[0])
print(matrix[1, 2])
print(matrix @ np.array([1, 2, 3]))
