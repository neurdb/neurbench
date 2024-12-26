import itertools
import time
from typing import Generator, List, Sequence
import typing
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .deterministic import sample_rng


def split_into_batches(lst, batch_size):
    """
    Splits a list into batches of a given size.

    Parameters:
        lst (list): The list to be split.
        batch_size (int): The size of each batch.

    Returns:
        list: A list of batches.
    """
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def para_sample(values: Generator, random_state: int):
    rng = np.random.default_rng(np.random.MT19937(seed=random_state))

    result = []

    for v in values:
        if not issubclass(v.__class__, pd.Interval):
            result.append(v)
            continue

        if isinstance(v.left, float):
            result.append(rng.uniform(v.left, v.right))
        elif isinstance(v.left, int):
            result.append(rng.integers(v.left, v.right))
        elif isinstance(v.left, np.integer):
            result.append(rng.integers(int(v.left), int(v.right)))
        else:
            raise ValueError(f"Unsupported value type: {type(v.left)}")

    return result


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
    start_time = time.time()
    sampled_indices = sample_rng.choice(
        len(distribution), size=num_samples, p=distribution
    )
    print(f"Sampling time: {time.time() - start_time}")

    # Get the corresponding values for the sampled indices
    start_time = time.time()
    
    if len(values) < 32768:
        """Sequential Sampling"""
        result = para_sample((values[i] for i in sampled_indices), 1)
    
    else:
        """Parallel Sampling"""
        result = list(
            itertools.chain.from_iterable(
                typing.cast(
                    List,
                    Parallel(n_jobs=-1, batch_size=8192, verbose=10)(
                        delayed(para_sample)(b, i+1)
                        for i, b in enumerate(split_into_batches(sampled_indices, 8192))
                    ),
                )
            )
        )

    print(f"Reassigning time: {time.time() - start_time}")

    return result, sampled_indices


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
