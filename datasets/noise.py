import numpy as np

def add_label_noise(
    targets,
    noise_ratio,
    seed=None,
):
    targets_array = np.asarray(targets)

    if targets_array.ndim != 1:
        raise ValueError("array must have 1 dim")

    if noise_ratio > 1 or noise_ratio < 0:
        raise ValueError("noise_ratio must be between 0 and 1")

    noisy_targets = np.copy(targets_array)
    noise_count = int(len(targets_array) * noise_ratio)

    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(targets_array), size=noise_count, replace=False)
    noisy_targets[selected_indices] = 1 - noisy_targets[selected_indices]

    return noisy_targets