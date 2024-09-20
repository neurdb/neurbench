import numpy as np
import pandas as pd


def categorical_dist(series: pd.Series):
    """get the distribution of the values in column column_index"""
    return series.value_counts(normalize=True)


def numerical_dist(series: pd.Series, n_bins: int):
    """get the distribution of numerical values, put them into n_bins bins"""
    return series.value_counts(bins=n_bins, normalize=True)


def numerical_dist_on_predefined_bins(series: pd.Series, bins: list):
    """get the distribution of numerical values in pre-defined bins"""
    indices = pd.IntervalIndex.from_tuples([(x["start"], x["end"]) for x in bins])
    return pd.cut(series, bins=indices).value_counts(normalize=True)

def categorical_dist_on_predefined_bins(series: pd.Series, bins: list):
    """get the distribution of categorical values in pre-defined bins"""
    distribution = series.value_counts().reindex(bins, fill_value=0)
    return distribution / len(series) * 100

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
    weights = 1.0 / np.power(np.arange(1, n+1), s)
    
    # Normalize to create a valid probability distribution
    return weights / np.sum(weights)