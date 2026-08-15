from split import train_test_split
import numpy as np

features_arr = np.random.rand(500, 2)
targets_arr = np.random.rand(500, )

features_train, features_test, targets_train, targets_test = train_test_split(
    features_arr,
    targets_arr,
    seed=5
)

assert len(features_train) == 400
assert len(features_test) == 100

assert features_train.shape[1] == features_test.shape[1]

assert targets_train.shape == (400,)
assert targets_test.shape == (100,)