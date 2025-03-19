import deterministic

deterministic.seed_everything(42)

import argparse
import os
from typing import List, Literal

import numpy as np
import pandas as pd
from scipy.spatial import distance
import data_utils as du

CORR_TYPES: List[Literal["pearson", "kendall", "spearman"]] = ["pearson", "spearman"]


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-name", type=str, default="imdb")
parser.add_argument("--table-name", type=str, default="nosuchtable")
parser.add_argument("--expdir", type=str, default="expdir")
parser.add_argument("--enable-corr", action="store_true", default=False)
parser.add_argument("--enable-drift", action="store_true", default=False)
parser.add_argument("--variant-id", type=int, default=-1)

args = parser.parse_args()


# dataset_path = os.path.join(
#     "datasets", "minority_class_oversampling", f"{dataset_name}_train.csv"
# )

dataset_path = os.path.join("datasets", args.dataset_name, f"{args.table_name}.csv")

# save_dir = os.path.join(args.expdir, f"{dataset_name}_output")
if args.variant_id > 0:
    save_dir = os.path.join(
        args.expdir, args.dataset_name, f"{args.table_name}-{args.variant_id}"
    )
else:
    save_dir = os.path.join(args.expdir, args.dataset_name, args.table_name)

assert os.path.exists(save_dir)

drifted_data_path = os.path.join(save_dir, f"{args.table_name}.drifted.csv")

original_data = pd.read_csv(
    dataset_path, doublequote=False, escapechar="\\", low_memory=False
)
print(original_data.head())
drifted_data = pd.read_csv(
    drifted_data_path, doublequote=False, escapechar="\\", low_memory=False
)
print(drifted_data.head())

if args.enable_corr:
    # combined_data = pd.concat((original_data, drifted_data))

    for corr_type in CORR_TYPES:
        original_corr = original_data.corr(method=corr_type, numeric_only=True)
        drifted_corr = drifted_data.corr(method=corr_type, numeric_only=True)

        print(f"[original] {corr_type} correlation: ")
        print(original_corr)
        print(f"[drifted] {corr_type} correlation: ")
        print(drifted_corr)

        loss = (drifted_corr - original_corr).abs()
        mean_abs_loss = loss.mean().mean()
        sum_abs_loss = loss.sum().sum()
        print(
            f"[{corr_type}] mean absolute loss: {mean_abs_loss:.6f}, sum absolute loss: {sum_abs_loss:.6f}"
        )
        print()


def numerical_dist(series: pd.Series, n_bins: int):
    """Get the distribution of numerical values, put them into n_bins bins."""
    # if series values are integers, return bins of integers
    if all(isinstance(x, int) for x in series):
        print("all values are integers")
        edges = np.linspace(series.min(), series.max(), n_bins + 1)
        edges = np.unique(edges.astype(int))  # Ensure unique bin edges

        # Check if the number of unique edges is less than n_bins
        if len(edges) <= n_bins:
            raise ValueError(
                "The number of unique bin edges is less than the number of requested bins."
            )

        result = pd.cut(
            series, bins=edges, precision=0, include_lowest=True
        ).value_counts(normalize=True, sort=False)

        return result
    else:
        return series.value_counts(bins=n_bins, normalize=True, sort=False)


def numerical_dist_on_predefined_bins(series: pd.Series, bins: list):
    """get the distribution of numerical values in pre-defined bins"""
    indices = pd.IntervalIndex.from_tuples([(x["start"], x["end"]) for x in bins])
    return pd.cut(series, bins=indices).value_counts(normalize=True, sort=False)


def categorical_dist(series: pd.Series):
    """get the distribution of the values in column column_index"""
    return series.value_counts(normalize=True)


def categorical_dist_on_predefined_bins(series: pd.Series, bins: list):
    """get the distribution of categorical values in pre-defined bins"""
    # print(bins)
    # print(series)
    distribution = series.value_counts().reindex(bins, fill_value=0)
    return distribution / len(series)


def is_numerical_column(series: pd.Series, threshold: int = 1):
    # if there is float, => numerical
    for x in series:
        if isinstance(x, float):
            print("found float")
            return True

        # if isinstance(x, (str, object)):
        #     print("found string")
        #     return False

    # if all int + (unique values < thresholw) => cate
    if all(isinstance(x, int) for x in series):
        unique_count = series.nunique()
        if unique_count < threshold:
            return False
        else:
            return True

    # If the series contains non-integer/non-float types, return False by default
    return False


if args.enable_drift:
    """get drifted columns"""
    dataset_info = du.load_json(f"datasets/{args.dataset_name}/dataset_info.json")
    dataset_info = dataset_info[args.table_name]
    drifted_columns = dataset_info["applicable_columns"]

    """compute drift of drifted data from original data"""
    # original_data = original_data.select_dtypes(exclude=["object"])
    # drifted_data = drifted_data.select_dtypes(exclude=["object"])

    divergences = []

    for col in drifted_columns:
        col_data = original_data[col].dropna()

        if is_numerical_column(col_data):
            print("processing numerical column", col)

            if all(isinstance(x, int) for x in col_data):
                nunique = col_data.nunique()

                original_col = numerical_dist(
                    col_data, nunique - 1 if nunique < 20 else 20
                )
            else:
                original_col = numerical_dist(col_data, 20)

            bins = sorted(
                [{"start": x.left, "end": x.right} for x in original_col.index],
                key=lambda x: x["start"],
            )
            drifted_col = numerical_dist_on_predefined_bins(drifted_data[col], bins)
        else:
            print("processing categorical column", col)

            original_col = categorical_dist(col_data)
            print("original col:")
            print(original_col.reset_index().sort_values(by="index"))
            bins = sorted(original_col.index)
            drifted_col = categorical_dist_on_predefined_bins(drifted_data[col], bins)
            print("drifted col:")
            print(drifted_col.reset_index().sort_values(by="index"))

        # print("original_col", original_col)
        # print("drifted_col", drifted_col)

        jsd = distance.jensenshannon(original_col, drifted_col)
        if np.isnan(jsd):
            jsd = 1.0

        print(f"JS divergence [{col}]: {jsd}")
        # print()

        divergences.append(jsd)

    print(f"mean JS divergence: {np.mean(divergences)}")
