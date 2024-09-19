import pandas as pd


def categorical_dist(series: pd.Series):
    """get the distribution of the values in column column_index"""
    return series.value_counts(normalize=True)


def numerical_dist(series: pd.Series, n_bins: int):
    """get the distribution of numerical values, put them into n_bins bins"""
    return series.value_counts(bins=n_bins, normalize=True)
