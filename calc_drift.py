"""
Calculate drift and correlation between two CSV files or datasets.
Reuses logic from postproc.py.

Usage:
    # Single table
    python calc_drift.py --file1 datasets/imdb/aka_title.csv --file2 datasets/imdb_2017/aka_title.csv --dataset-name imdb --table-name aka_title

    # All tables between two datasets
    python calc_drift.py --src-dataset imdb --dst-dataset imdb_2017 --output drift_ref.csv

This calculates:
1. JS divergence (drift) between file1 and file2
2. Correlation difference between file1 and file2
"""

import argparse
import os
import sys

# Add parent directory to path to import from postproc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.spatial import distance
import drift_ddpm.data_utils as du

CORR_TYPES = ["pearson", "spearman"]


def load_csv(path: str) -> pd.DataFrame:
    """
    Smart CSV loader that auto-detects the correct quoting strategy.
    If first strategy has warnings/skipped lines, try next one.
    """
    import warnings

    # Use escapechar first for: .drifted.csv files OR imdb dataset (without year)
    use_escapechar_first = path.endswith(".drifted.csv") or "/imdb/" in path or "\\imdb\\" in path

    if use_escapechar_first:
        strategies = [
            {"doublequote": False, "escapechar": "\\"},   # Backslash escaped
            {"doublequote": True},                        # Standard CSV (fallback)
        ]
    else:
        strategies = [
            {"doublequote": True},                        # Standard CSV (most common)
            {"doublequote": False, "escapechar": "\\"},   # Backslash escaped
        ]

    last_df = None
    for strategy in strategies:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                df = pd.read_csv(path, low_memory=False, on_bad_lines='warn', **strategy)

            # If no warnings, this strategy works - return immediately
            if len(w) == 0:
                return df
            # Otherwise save and try next strategy
            last_df = df
        except:
            continue

    # All strategies had warnings, return last successful one
    if last_df is not None:
        return last_df
    raise RuntimeError(f"Failed to load {path}")


def numerical_dist(series: pd.Series, n_bins: int):
    """Get the distribution of numerical values, put them into n_bins bins."""
    if all(isinstance(x, int) for x in series):
        edges = np.linspace(series.min(), series.max(), n_bins + 1)
        edges = np.unique(edges.astype(np.int64))
        if len(edges) <= n_bins:
            raise ValueError(
                "The number of unique bin edges is less than the number of requested bins."
            )
        return pd.cut(series, bins=edges, precision=0, include_lowest=True).value_counts(normalize=True, sort=False)
    else:
        return series.value_counts(bins=n_bins, normalize=True, sort=False)


def numerical_dist_on_predefined_bins(series: pd.Series, bins: list):
    indices = pd.IntervalIndex.from_tuples([(x["start"], x["end"]) for x in bins])
    return pd.cut(series, bins=indices).value_counts(normalize=True, sort=False)


def categorical_dist(series: pd.Series):
    return series.value_counts(normalize=True)


def categorical_dist_on_predefined_bins(series: pd.Series, bins: list):
    distribution = series.value_counts().reindex(bins, fill_value=0)
    return distribution / len(series)


def is_numerical_column(series: pd.Series, threshold: int = 1):
    # if there is PYTHON float (not numpy.float64), => numerical
    for x in series:
        if isinstance(x, float):
            return True

    # if all PYTHON int (not numpy.int64) + (unique values >= threshold) => numerical
    if all(isinstance(x, int) for x in series):
        unique_count = series.nunique()
        if unique_count < threshold:
            return False
        else:
            return True

    # If the series contains non-integer/non-float types (including numpy.int64), return False
    return False


def calc_correlation(df1: pd.DataFrame, df2: pd.DataFrame, verbose: bool = True, exclude_columns: list = None):
    """Calculate correlation difference between two dataframes.

    Args:
        df1: First dataframe (base/original)
        df2: Second dataframe (generated/target)
        verbose: Whether to print detailed output
        exclude_columns: Columns to exclude from correlation (default: ['id'])

    Returns dict with keys:
        - 'pearson': mean absolute loss for pearson correlation
        - 'spearman': mean absolute loss for spearman correlation
        - 'pearson_abs': mean absolute correlation value of df1 (base)
        - 'spearman_abs': mean absolute correlation value of df1 (base)
        - 'pearson_abs_gen': mean absolute correlation value of df2 (generated)
        - 'spearman_abs_gen': mean absolute correlation value of df2 (generated)
    """
    # Default: exclude 'id' column (auto-incrementing, creates pseudo-correlation)
    if exclude_columns is None:
        exclude_columns = ['id']

    # Filter out excluded columns
    df1_filtered = df1.drop(columns=[c for c in exclude_columns if c in df1.columns], errors='ignore')
    df2_filtered = df2.drop(columns=[c for c in exclude_columns if c in df2.columns], errors='ignore')

    if verbose:
        excluded = [c for c in exclude_columns if c in df1.columns]
        if excluded:
            print(f"\n[Excluding columns from correlation: {excluded}]")
        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)

    results = {}
    for corr_type in CORR_TYPES:
        corr1 = df1_filtered.corr(method=corr_type, numeric_only=True)
        corr2 = df2_filtered.corr(method=corr_type, numeric_only=True)

        loss = (corr2 - corr1).abs()
        mean_abs_loss = loss.mean().mean()
        results[corr_type] = mean_abs_loss

        # Calculate mean absolute correlation values (excluding diagonal)
        # Fill diagonal with 0 to exclude self-correlation (always 1)
        corr1_no_diag = corr1.copy()
        corr2_no_diag = corr2.copy()
        np.fill_diagonal(corr1_no_diag.values, 0)
        np.fill_diagonal(corr2_no_diag.values, 0)

        abs_corr1 = corr1_no_diag.abs().mean().mean()
        abs_corr2 = corr2_no_diag.abs().mean().mean()
        results[f'{corr_type}_abs'] = abs_corr1
        results[f'{corr_type}_abs_gen'] = abs_corr2

        if verbose:
            ratio = mean_abs_loss / abs_corr1 if abs_corr1 > 0 else 0
            print(f"\n[{corr_type}] mean absolute loss: {mean_abs_loss:.6f}, "
                  f"abs_corr(base)={abs_corr1:.4f}, abs_corr(gen)={abs_corr2:.4f}, "
                  f"loss/abs_ratio={ratio:.4f}")

            # Print per-column-pair correlation details
            cols = corr1.columns.tolist()
            print(f"  {'col_i':<20} {'col_j':<20} {'base':>10} {'gen':>10} {'diff':>10}")
            print(f"  {'-'*20} {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
            for i, col_i in enumerate(cols):
                for j, col_j in enumerate(cols):
                    if i < j:  # Only upper triangle (exclude diagonal and duplicates)
                        base_val = corr1.loc[col_i, col_j]
                        gen_val = corr2.loc[col_i, col_j] if col_j in corr2.columns and col_i in corr2.index else float('nan')
                        diff = abs(gen_val - base_val) if not np.isnan(gen_val) else float('nan')
                        print(f"  {col_i:<20} {col_j:<20} {base_val:>10.4f} {gen_val:>10.4f} {diff:>10.4f}")

    return results


def calc_drift(df1: pd.DataFrame, df2: pd.DataFrame, columns: list, verbose: bool = True):
    """Calculate JS divergence (drift) between two dataframes.

    Same logic as postproc.py:
    - is_numerical_column True (float or Python int): use interval binning
    - is_numerical_column False (numpy.int64/FK columns, strings): use unique values
    """
    if verbose:
        print("\n" + "=" * 60)
        print("DRIFT ANALYSIS (JS Divergence)")
        print("=" * 60)

    divergences = []

    for col in columns:
        if col not in df1.columns or col not in df2.columns:
            if verbose:
                print(f"[{col}] Warning: column not found, skipping")
            continue

        col_data = df1[col].dropna()
        if len(col_data) == 0:
            continue
        # , threshold=len(col_data)):
        try:
            if is_numerical_column(col_data):
                if verbose:
                    print(f"processing numerical column {col}")

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
                drifted_col = numerical_dist_on_predefined_bins(df2[col], bins)
            else:
                if verbose:
                    print(f"processing categorical column {col}")

                original_col = categorical_dist(col_data)
                bins = sorted(original_col.index)
                drifted_col = categorical_dist_on_predefined_bins(df2[col], bins)
                # Align original_col to bins order (categorical_dist returns freq-descending order)
                original_col = original_col.reindex(bins)

            jsd = distance.jensenshannon(original_col, drifted_col)
            if np.isnan(jsd):
                jsd = 1.0

            if verbose:
                print(f"[{col}] JS divergence: {jsd:.6f}")
            divergences.append(jsd)
        except Exception as e:
            if verbose:
                print(f"[{col}] Error: {e}")
            continue

    if divergences:
        mean_drift = np.mean(divergences)
        if verbose:
            print(f"\nMean JS divergence: {mean_drift:.6f}")
        return mean_drift
    return 0.0


def _generate_dataset_info(src_dir: str) -> dict:
    """Auto-generate dataset_info.json by copying structure from base dataset and updating n_samples."""
    import json

    # Detect dataset type from directory name
    src_name = os.path.basename(src_dir.rstrip('/'))
    if 'stack' in src_name.lower():
        template_path = "datasets/stack/dataset_info.json"
        template_name = "stack"
    else:
        template_path = "datasets/imdb/dataset_info.json"
        template_name = "imdb"

    if not os.path.exists(template_path):
        print(f"Error: Template {template_path} not found")
        return {}

    with open(template_path) as f:
        template = json.load(f)

    dataset_info = {}
    print(f"Auto-generating dataset_info.json using {template_name} as template...")

    for table_name, table_config in template.items():
        csv_path = f"{src_dir}/{table_name}.csv"

        if table_config is None:
            # Keep null entries as-is
            dataset_info[table_name] = None
            continue

        if not os.path.exists(csv_path):
            print(f"  {table_name}: CSV not found, skipped")
            dataset_info[table_name] = None
            continue

        try:
            # Count rows
            with open(csv_path, 'r') as f:
                n_samples = sum(1 for _ in f) - 1  # minus header

            dataset_info[table_name] = {
                "applicable_columns": table_config["applicable_columns"],
                "n_samples": n_samples
            }
            print(f"  {table_name}: {n_samples} rows")
        except Exception as e:
            print(f"  {table_name}: error - {e}")
            dataset_info[table_name] = None

    return dataset_info


def calc_all_tables(src_dir: str, dst_dir: str, output: str):
    """Calculate drift and correlation for all driftable tables between two datasets."""
    import json

    # Load or generate dataset_info.json
    info_path = f"{src_dir}/dataset_info.json"
    if not os.path.exists(info_path):
        print(f"dataset_info.json not found, auto-generating...")
        dataset_info = _generate_dataset_info(src_dir)

        # Save generated dataset_info.json
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=4)
        print(f"Saved auto-generated dataset_info.json to {info_path}\n")
    else:
        with open(info_path) as f:
            dataset_info = json.load(f)

    # Get driftable tables (non-null entries)
    driftable_tables = {k: v for k, v in dataset_info.items() if v is not None}

    # Extract dataset names from directory paths
    src_dataset = os.path.basename(src_dir.rstrip('/'))
    dst_dataset = os.path.basename(dst_dir.rstrip('/'))

    print(f"Source: {src_dir} ({src_dataset})")
    print(f"Destination: {dst_dir} ({dst_dataset})")
    print(f"Driftable tables: {list(driftable_tables.keys())}")
    print(f"Output: {output}")
    print("=" * 70)

    results = []

    for table_name, table_info in driftable_tables.items():
        src_path = f"{src_dir}/{table_name}.csv"
        dst_path = f"{dst_dir}/{table_name}.csv"

        if not os.path.exists(src_path):
            print(f"[{table_name}] Source not found: {src_path}")
            continue
        if not os.path.exists(dst_path):
            print(f"[{table_name}] Destination not found: {dst_path}")
            continue

        columns = table_info.get("applicable_columns", [])
        print(f"[{table_name}] Processing... (columns: {columns})")

        try:
            df1 = load_csv(src_path)
            df2 = load_csv(dst_path)

            # Calculate
            corr_results = calc_correlation(df1, df2, verbose=False)
            drift = calc_drift(df1, df2, columns, verbose=False)

            pearson = corr_results.get("pearson", None)
            spearman = corr_results.get("spearman", None)

            drift_str = f"{drift:.4f}" if drift is not None else "N/A"
            pearson_str = f"{pearson:.4f}" if pearson is not None and not np.isnan(pearson) else "N/A"
            print(f"  Drift: {drift_str}, Pearson: {pearson_str}")

            results.append({
                "table_name": table_name,
                "columns": ",".join(columns),
                "drift": drift,
                "pearson_corr_loss": pearson,
                "spearman_corr_loss": spearman,
                "src_dataset": src_dataset,
                "dst_dataset": dst_dataset,
            })
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "table_name": table_name,
                "columns": ",".join(columns),
                "drift": None,
                "pearson_corr_loss": None,
                "spearman_corr_loss": None,
                "src_dataset": src_dataset,
                "dst_dataset": dst_dataset,
            })

    # Save to CSV (append if exists)
    df_new = pd.DataFrame(results)

    if os.path.exists(output) and os.path.getsize(output) > 0:
        # Load existing and append
        try:
            df_existing = pd.read_csv(output)
        except pd.errors.EmptyDataError:
            df_existing = pd.DataFrame()
        # Remove duplicates (same src_dataset, dst_dataset, table_name)
        if not df_existing.empty and "src_dataset" in df_existing.columns:
            for _, row in df_new.iterrows():
                mask = (
                    (df_existing["src_dataset"] == row["src_dataset"]) &
                    (df_existing["dst_dataset"] == row["dst_dataset"]) &
                    (df_existing["table_name"] == row["table_name"])
                )
                df_existing = df_existing[~mask]
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new
        print(f"\nAppended to existing file: {output}")
    else:
        df = df_new
        print(f"\nCreated new file: {output}")

    df.to_csv(output, index=False)

    print("=" * 70)
    print("\nNew results:")
    print(df_new.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Calculate drift and correlation between two CSV files or datasets")

    # Single file mode
    parser.add_argument("--file1", type=str, default=None, help="First CSV file (source)")
    parser.add_argument("--file2", type=str, default=None, help="Second CSV file (target)")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name for loading applicable_columns")
    parser.add_argument("--table-name", type=str, default=None, help="Table name for loading applicable_columns")
    parser.add_argument("--columns", type=str, default=None, help="Comma-separated columns (overrides dataset_info)")

    # Batch mode
    parser.add_argument("--src-dir", type=str, default=None, help="Source dataset directory (batch mode)")
    parser.add_argument("--dst-dir", type=str, default=None, help="Destination dataset directory (batch mode)")
    parser.add_argument("--output", type=str, default="drift_ref.csv", help="Output CSV file (batch mode)")

    args = parser.parse_args()

    # Batch mode: process all tables
    if args.src_dir and args.dst_dir:
        calc_all_tables(args.src_dir, args.dst_dir, args.output)
        return

    # Single file mode
    if not args.file1 or not args.file2:
        print("Error: Either provide --file1 and --file2, or --src-dataset and --dst-dataset")
        parser.print_help()
        return

    # Load data
    print(f"File 1: {args.file1}")
    df1 = load_csv(args.file1)
    print(f"  Shape: {df1.shape}")

    print(f"File 2: {args.file2}")
    df2 = load_csv(args.file2)
    print(f"  Shape: {df2.shape}")

    # Get columns
    if args.columns:
        columns = [c.strip() for c in args.columns.split(",")]
    elif args.dataset_name and args.table_name:
        info_path = f"datasets/{args.dataset_name}/dataset_info.json"
        if os.path.exists(info_path):
            info = du.load_json(info_path)
            if args.table_name in info and info[args.table_name]:
                columns = info[args.table_name].get("applicable_columns", [])
                print(f"Using columns from dataset_info.json: {columns}")
            else:
                columns = df1.select_dtypes(include=[np.number]).columns.tolist()
        else:
            columns = df1.select_dtypes(include=[np.number]).columns.tolist()
    else:
        columns = df1.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Auto-detected columns: {columns}")

    # Calculate
    corr_result = calc_correlation(df1, df2)
    mean_drift = calc_drift(df1, df2, columns)

    print(f"\nJS Divergence: {mean_drift:.4f}, Pearson Corr Loss: {corr_result['pearson']:.4f}")


if __name__ == "__main__":
    main()
