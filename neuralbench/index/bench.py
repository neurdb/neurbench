import random
import numpy as np
from scipy import stats
from collections import defaultdict

from typing import List, Tuple, Optional, Dict


def sample_bulkloading_keyset(
        bin_idxs: np.ndarray,
        distribution: np.ndarray,
        bin_to_keyset: Dict,
        num_keys: int,
        verbose: bool = False) -> List[int]:
    """
    sample keys from each bin according to the distribution
    :param bin_idxs: bin indices
    :param distribution: distribution of each bin
    :param bin_to_keyset: mapping from bin index to key set
    :param num_keys: total number of keys to sample
    :param verbose: whether to print out the exceeding bins
    :return: a list of keys
    """
    bulk_loading_keys = []
    exceeding_cnt = 0
    assert len(bin_idxs) == len(distribution)
    for bin_idx, p in zip(bin_idxs, distribution):
        sample_n = int(p * num_keys)
        waiting_sample_keys = bin_to_keyset[bin_idx]
        bound_n = min(sample_n, len(waiting_sample_keys))

        if sample_n > bound_n:
            exceeding_cnt += 1
            if verbose:
                print(
                    f"Exceeding the number of keys in bin {bin_idx}, sample_n: {sample_n}, bound_n: {bound_n}")
            sample_n = bound_n

        keys = random.sample(waiting_sample_keys, sample_n)
        bulk_loading_keys.extend(keys)

    if verbose:
        print(f"Exceeding bins: {exceeding_cnt} / {len(bin_idxs)}")

    return bulk_loading_keys


class KeySetBinEncoder(object):
    UINT32_DEFAULT_OFFSET = 20
    UINT64_DEFAULT_OFFSET = 40

    @staticmethod
    def bin_keyset_to_distribution(
        key_set: np.ndarray,
        bin_width_offset: Optional[int],
        verbose: bool = False,
        details: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Bin the key set into bins and compute the probability distribution.
        the bin_with_offset should be heuristically set
        :param key_set: The key set to bin
        :param bin_width_offset: The offset to use for binning
        :param verbose: Whether to print out the default bin width offset
        :param details: Whether to return the details of the binning
        :return: Tuple
        :   - bin_idxs: The bin indices
        :   - probability_distribution: The probability distribution
        :   - bin_idx_to_key: A dictionary mapping bin indices to keys"""
        if bin_width_offset is None:
            if key_set.dtype == np.uint32:
                bin_width_offset = KeySetBinEncoder.UINT32_DEFAULT_OFFSET
            elif key_set.dtype == np.uint64:
                bin_width_offset = KeySetBinEncoder.UINT64_DEFAULT_OFFSET
            else:
                raise ValueError("Invalid key type")
            if verbose:
                print(
                    f"Using default bin width offset {bin_width_offset} / {key_set.dtype}")

        bin_idx_for_each_key = key_set >> bin_width_offset
        bin_idxs, counts = np.unique(
            bin_idx_for_each_key, return_counts=True)

        total_count = np.sum(counts)
        probability_distribution = counts / total_count

        bin_idx_to_key = defaultdict(list)
        if details:
            for key, bin_idx in zip(key_set, bin_idx_for_each_key):
                bin_idx_to_key[bin_idx].append(key)

        if verbose:
            print(f"Total number of bins: {len(bin_idxs)}")
        return bin_idxs, probability_distribution, bin_idx_to_key

    @staticmethod
    def plot_probability_distribution(probability_distribution: np.ndarray, x:Optional[List] = None) -> None:
        """
        Plot the probability distribution.
        :param probability_distribution: The probability distribution to plot
        """
        import matplotlib.pyplot as plt
        labels = x if x else range(len(probability_distribution))
        max_value = max(probability_distribution)
        rounded_max_value = np.ceil(max_value * 20) / 20

        plt.bar(labels, probability_distribution, color='blue', alpha=0.7)
        plt.xlabel('Categories')
        plt.ylabel('Probability')
        plt.title('Probability Distribution')
        plt.ylim(0, rounded_max_value)  # Set y-axis limits from 0 to 1
        plt.grid(axis='y')

        # Show the plot
        plt.show()

    @staticmethod
    def plot_cdf(data: np.ndarray,
                 title: Optional[str] = "Cumulative Distribution Function (CDF)",
                 min_value: Optional[int] = None,
                 max_value: Optional[int] = None,
                 figsize: Optional[Tuple] = (4, 3),
                 offset: Optional[int] = 1e18) -> None:
        """
        Plot the cumulative distribution function.
        :param data: The data to plot
        :param title: The title of the plot
        :param min_value: The minimum value to plot
        :param max_value: The maximum value to plot
        """
        import matplotlib.pyplot as plt
        sorted_key = np.sort(data)
        cdf = np.arange(1, len(sorted_key) + 1) / len(sorted_key)
        max_value = max_value if not max_value else np.max(data)
        min_value = min_value if not min_value else np.min(data)

        plt.figure(figsize=figsize)
        plt.plot(sorted_key, cdf)
        plt.xlabel('Data values')
        plt.ylabel('CDF')
        plt.xlim(min_value - offset, max_value + offset)
        plt.title(title)
        plt.grid(True)
        plt.show()

    ''' ---------------------- Abnormal Value Filter ----------------------'''
    @staticmethod
    def filter_abnormal_values(
            key_set: np.ndarray,
            method: str = "IQR",
            verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter out abnormal values from the key set using the specified method.
        :param key_set: The key set to filter
        :param method: The method to use for filtering. One of ["IQR", "CONFIDENCE"]
        :param verbose: Whether to print out the number of values filtered
        :return: A tuple containing the filtered key set and the removed key set
        """
        if method == "IQR":
            lower_bound, upper_bound = KeySetBinEncoder.__interquartile_range__(
                key_set)
        elif method == "CONFIDENCE":
            lower_bound, upper_bound = KeySetBinEncoder.__normal_confidence_interval__(
                key_set)
        else:
            raise ValueError("Invalid method")

        bit = (key_set >= lower_bound) & (key_set <= upper_bound)
        filtered_key = key_set[bit]
        removed_key = key_set[~bit]

        if verbose:
            print(
                f"Filtered {len(key_set) - len(filtered_key)} values, [{len(filtered_key)} / {len(key_set)}]")

        return filtered_key, removed_key

    @staticmethod
    def __normal_confidence_interval__(key_set: np.ndarray, confidence_level: float = 0.9545) -> Tuple[float, float]:
        """
        Compute the confidence interval for the key set using the normal distribution.
        :param key_set: The key set
        :param confidence_level: The confidence level
        :return: A tuple containing the lower and upper bounds of the confidence interval
        """
        mean = np.mean(key_set)
        std_dev = np.std(key_set)

        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        lower_bound = mean - z_score * std_dev
        upper_bound = mean + z_score * std_dev
        return lower_bound, upper_bound

    @staticmethod
    def __interquartile_range__(key_set: np.ndarray) -> Tuple[float, float]:
        """
        Compute the interquartile range for the key set.
        :param key_set: The key set
        :return: A tuple containing the lower and upper bounds of the interquartile range
        """

        q1 = np.percentile(key_set, 25)
        q3 = np.percentile(key_set, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return lower_bound, upper_bound
