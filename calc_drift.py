"""
Calculate drift and correlation between two CSV files or datasets.
Reuses logic from postproc.py.

Usage:
    # Single table
    python calc_drift.py --file1 datasets/imdb/aka_title.csv --file2 datasets/imdb_2017/aka_title.csv --dataset-name imdb --table-name aka_title

    # All tables between two datasets
    python calc_drift.py --src-dataset imdb --dst-dataset imdb_2017 --output drift_reference.csv

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
    """Load CSV with multiple strategy fallbacks."""
    for strategy in [
        {"doublequote": True},                        # Standard CSV (most common)
        {"doublequote": False, "escapechar": "\\"},   # Backslash escaped
    ]:
        try:
            return pd.read_csv(path, low_memory=False, on_bad_lines='skip', **strategy)
        except:
            continue
    raise RuntimeError(f"Failed to load {path}")


def numerical_dist(series: pd.Series, n_bins: int):
    if all(isinstance(x, int) for x in series.dropna()):
        edges = np.linspace(series.min(), series.max(), n_bins + 1)
        edges = np.unique(edges.astype(np.int64))
        if len(edges) <= n_bins:
            n_bins = len(edges) - 1
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
    for x in series.dropna():
        if isinstance(x, float):
            return True
    if all(isinstance(x, int) for x in series.dropna()):
        return series.nunique() >= threshold
    return False


def calc_correlation(df1: pd.DataFrame, df2: pd.DataFrame, verbose: bool = True):
    """Calculate correlation difference between two dataframes."""
    if verbose:
        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)

    results = {}
    for corr_type in CORR_TYPES:
        corr1 = df1.corr(method=corr_type, numeric_only=True)
        corr2 = df2.corr(method=corr_type, numeric_only=True)

        loss = (corr2 - corr1).abs()
        mean_abs_loss = loss.mean().mean()
        results[corr_type] = mean_abs_loss

        if verbose:
            print(f"[{corr_type}] mean absolute loss: {mean_abs_loss:.6f}")

    return results


def calc_drift(df1: pd.DataFrame, df2: pd.DataFrame, columns: list, verbose: bool = True):
    """Calculate JS divergence (drift) between two dataframes."""
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

        try:
            if is_numerical_column(col_data):
                nunique = col_data.nunique()
                n_bins = nunique - 1 if nunique < 20 else 20
                if n_bins < 2:
                    n_bins = 2

                original_col = numerical_dist(col_data, n_bins)
                bins = sorted(
                    [{"start": x.left, "end": x.right} for x in original_col.index if hasattr(x, 'left')],
                    key=lambda x: x["start"],
                )
                if not bins:
                    continue
                drifted_col = numerical_dist_on_predefined_bins(df2[col], bins)
            else:
                original_col = categorical_dist(col_data)
                bins = sorted(original_col.index)
                drifted_col = categorical_dist_on_predefined_bins(df2[col], bins)

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
    """Auto-generate dataset_info.json by copying structure from imdb and updating n_samples."""
    import json

    # Use imdb's dataset_info.json as template
    template_path = "datasets/imdb/dataset_info.json"
    if not os.path.exists(template_path):
        print(f"Error: Template {template_path} not found")
        return {}

    with open(template_path) as f:
        template = json.load(f)

    dataset_info = {}
    print(f"Auto-generating dataset_info.json using imdb as template...")

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

    if os.path.exists(output):
        # Load existing and append
        df_existing = pd.read_csv(output)
        # Remove duplicates (same src_dataset, dst_dataset, table_name)
        key_cols = ["src_dataset", "dst_dataset", "table_name"]
        for _, row in df_new.iterrows():
            mask = (
                (df_existing["src_dataset"] == row["src_dataset"]) &
                (df_existing["dst_dataset"] == row["dst_dataset"]) &
                (df_existing["table_name"] == row["table_name"])
            )
            df_existing = df_existing[~mask]
        df = pd.concat([df_existing, df_new], ignore_index=True)
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
    parser.add_argument("--output", type=str, default="drift_reference.csv", help="Output CSV file (batch mode)")

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
    calc_correlation(df1, df2)
    mean_drift = calc_drift(df1, df2, columns)
    print("\n" + "=" * 60)
    print(f"TARGET DRIFT VALUE: {mean_drift:.2f}")
    print(f"Use: gd {args.dataset_name or 'DATASET'} {args.table_name or 'TABLE'} {mean_drift:.2f} --auto")
    print("=" * 60)


if __name__ == "__main__":
    main()
