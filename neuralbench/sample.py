from typing import List
import numpy as np
import pandas as pd
from .deterministic import sample_rng


def sample_from_distribution(distribution: List[float], values: list, num_samples: int):
    """
    Sample data from a given distribution and pair it with corresponding values.

    Args:
    distribution (array-like): The probability distribution to sample from.
    values (array-like): The corresponding values for each probability in the distribution.
    num_samples (int): The number of samples to generate.

    Returns:
    tuple: (sampled_values, sampled_indices)
        sampled_values: The sampled corresponding values.
        sampled_indices: The indices of the sampled items (for reference).
    """
    if len(distribution) != len(values):
        raise ValueError("The length of distribution and values must be the same.")

    # Ensure the distribution sums to 1
    distribution = np.array(distribution) / np.sum(distribution)

    # Sample indices based on the distribution
    sampled_indices = sample_rng.choice(
        len(distribution), size=num_samples, p=distribution
    )

    # Get the corresponding values for the sampled indices
    sampled_values = [values[i] for i in sampled_indices]

    for i in range(len(sampled_values)):
        v = sampled_values[i]
        if issubclass(v.__class__, pd.Interval):
            if isinstance(v.left, float):
                sampled_values[i] = sample_rng.uniform(v.left, v.right)
            elif isinstance(v.left, int):
                sampled_values[i] = sample_rng.integers(v.left, v.right)
            elif isinstance(v.left, np.integer):
                sampled_values[i] = sample_rng.integers(int(v.left), int(v.right))
            else:
                raise ValueError(f"Unsupported value type: {type(v.left)}")

    return sampled_values, sampled_indices


if __name__ == "__main__":
    distribution = [0.1, 0.2, 0.3, 0.4]
    values = ["a", "b", "c", "d"]
    num_samples = 1000
    sampled_values, sampled_indices = sample_from_distribution(
        distribution, values, num_samples
    )

    # print(sampled_values)
    print(sampled_indices)
    count = np.unique(sampled_indices, return_counts=True)[1] / num_samples
    print(count)
