from datasets.noise import add_label_noise
import numpy as np

targets = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
noisy_targets = add_label_noise(
    targets,
    noise_ratio=0.3,
    seed=42,
)

assert noisy_targets.sum() == 3
assert targets.sum() == 0
