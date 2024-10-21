import numpy as np
import pandas as pd


def categorical_dist(series: pd.Series):
    """get the distribution of the values in column column_index"""
    return series.value_counts(normalize=True)


def numerical_dist(series: pd.Series, n_bins: int):
    """Get the distribution of numerical values, put them into n_bins bins."""
    # if series values are integers, return bins of integers
    if all(isinstance(x, int) for x in series):
        print("all values are integers")
        edges = np.linspace(series.min(), series.max(), n_bins + 1)
        edges = np.unique(edges.astype(int))  # Ensure unique bin edges

        # Check if the number of unique edges is less than n_bins
        if len(edges) <= n_bins:
            raise ValueError("The number of unique bin edges is less than the number of requested bins.")

        result = pd.cut(series, bins=edges, precision=0, include_lowest=True).value_counts(
            normalize=True)

        return result
    else:
        return series.value_counts(bins=n_bins, normalize=True)


def numerical_dist_ori(series: pd.Series, n_bins: int):
    """get the distribution of numerical values, put them into n_bins bins"""
    # if series values are integers, return bins of integers
    if all(isinstance(x, int) for x in series):
        print("all values are integers")
        edges = np.linspace(series.min(), series.max(), n_bins + 1).astype(int)
        labels = [edges[i + 1] for i in range(n_bins)]
        result = pd.cut(series, bins=labels, precision=0, include_lowest=True).value_counts(normalize=True)
        # result.index = result.index.map(lambda x: pd.Interval(round(x.left), round(x.right))).

        return result
    else:
        return series.value_counts(bins=n_bins, normalize=True)


def numerical_dist_on_predefined_bins(series: pd.Series, bins: list):
    """get the distribution of numerical values in pre-defined bins"""
    indices = pd.IntervalIndex.from_tuples([(x["start"], x["end"]) for x in bins])
    return pd.cut(series, bins=indices).value_counts(normalize=True)


def categorical_dist_on_predefined_bins(series: pd.Series, bins: list):
    """get the distribution of categorical values in pre-defined bins"""
    print(bins)
    print(series)
    distribution = series.value_counts().reindex(bins, fill_value=0)
    return distribution / len(series)


def zipf_dist(n, s=1.0):
    """
    Generate a Zipf distribution of length n.
    
    Args:
    n (int): The length of the distribution.
    s (float): The value of the exponent characterizing the distribution. Default is 1.0.
    
    Returns:
    numpy.ndarray: An array of length n containing the Zipf distribution.
    """
    # Generate raw weights
    weights = 1.0 / np.power(np.arange(1, n + 1), s)

    # Normalize to create a valid probability distribution
    return weights / np.sum(weights)
