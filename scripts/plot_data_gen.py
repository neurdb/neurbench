#!/usr/bin/env python3
"""
Plot data generation quality comparison charts.
Supports task1 (IMDB-17 vs IMDB-17*) and task2 (IMDB-15 vs IMDB-15*).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import sys
import glob

sys.path.insert(0, ".")
import calc_drift

DRIFT_REF_FILE = "drift_ref.csv"
EXPDIR = "expdir"
DEFAULT_EXCLUDE_TABLES = {"title", "aka_name"}
CACHE_FILE = "plot_cache.json"


def load_cache():
    """Load cached data from file."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache_data):
    """Save cache data to file."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"Cache saved to {CACHE_FILE}")


def compute_all_data():
    """
    Compute all data needed for plotting and return as dict.
    This is the slow part that reads CSVs and calculates drift/correlation.
    """
    print("=" * 70)
    print("Computing all data (this may take a while)...")
    print("=" * 70)

    cache = {}

    # Task 1 & 3: Real values
    print("\n[1/6] Computing real drift/corr for imdb -> imdb_2017...")
    cache['real_imdb_2017'] = get_real_drift_corr('imdb', 'imdb_2017')

    print("\n[2/6] Computing real drift/corr for imdb -> imdb_2015...")
    cache['real_imdb_2015'] = get_real_drift_corr('imdb', 'imdb_2015')

    # Task 1: Generated values for imdb_2017
    print("\n[3/6] Computing generated drift/corr for ref=imdb_2017...")
    cache['gen_imdb_2017'] = get_generated_drift_corr('imdb_2017', 'imdb')

    # Task 3: Generated values for imdb_2015
    print("\n[4/6] Computing generated drift/corr for ref=imdb_2015...")
    cache['gen_imdb_2015'] = get_generated_drift_corr('imdb_2015', 'imdb')

    # Task 2: Drift experiments (0.1, 0.3, 0.5)
    print("\n[5/6] Computing drift experiment results (0.1, 0.3, 0.5)...")
    cache['drift_exp'] = {}
    for dv, vid in [(0.1, 1), (0.3, 3), (0.5, 5)]:
        print(f"  - drift={dv} (variant_id={vid})")
        cache['drift_exp'][str(dv)] = get_drift_exp_results(vid, 'imdb')

    # Task 1-cmp: Filtered generated values
    print("\n[6/6] Computing filtered generated data for base comparison...")
    all_bases = {"imdb", "imdb_2013", "imdb_2014", "imdb_2015", "imdb_2016"}
    no_13_14 = {"imdb", "imdb_2015", "imdb_2016"}
    no_15_16 = {"imdb", "imdb_2013", "imdb_2014"}

    cache['gen_filtered'] = {}
    for label, bases in [("all", all_bases), ("no_13_14", no_13_14), ("no_15_16", no_15_16)]:
        print(f"  - {label}: {bases}")
        cache['gen_filtered'][label] = get_generated_drift_corr_filtered('imdb_2017', 'imdb', bases)

    print("\n" + "=" * 70)
    print("All data computed!")

    return cache


def get_real_drift_corr(src_dataset: str, dst_dataset: str):
    """Get real drift and correlation from drift_ref.csv."""
    df = pd.read_csv(DRIFT_REF_FILE)
    mask = (df['src_dataset'] == src_dataset) & (df['dst_dataset'] == dst_dataset)
    subset = df[mask]

    result = {}
    for _, row in subset.iterrows():
        table = row['table_name']
        result[table] = {
            'drift': row['drift'],
            'corr': row['pearson_corr_loss']
        }
    return result


def get_generated_drift_corr(ref_dataset: str, compare_with: str):
    """
    Get best generated drift and correlation from all experiments.
    For each table, picks the experiment with smallest drift_diff and corr_diff separately.
    """
    # Load dataset_info for columns
    config_path = f"datasets/{compare_with}/dataset_info.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            dataset_config = json.load(f)
    else:
        dataset_config = {}

    # Find all matching experiment directories
    pattern = f"*_ref_{ref_dataset}"
    exp_dirs = sorted(glob.glob(os.path.join(EXPDIR, pattern)))

    # Collect all results per table
    all_results = {}  # {table: [entry, ...]}

    # Get target values
    real_values = get_real_drift_corr(compare_with, ref_dataset)

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        base = exp_name.replace(f"_ref_{ref_dataset}", "")

        tables = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]

        for table in tables:
            gen_csv = os.path.join(exp_dir, table, f"{table}.drifted.csv")

            if not os.path.exists(gen_csv):
                continue

            # Get target drift/corr
            target = real_values.get(table, {})
            target_drift = target.get('drift', 0)
            target_corr = target.get('corr', 0)

            # Calculate drift against compare_with
            orig_csv = f"datasets/{compare_with}/{table}.csv"
            if not os.path.exists(orig_csv):
                continue

            try:
                orig_df = calc_drift.load_csv(orig_csv)
                gen_df = calc_drift.load_csv(gen_csv)
                columns = dataset_config.get(table, {}).get("applicable_columns", [])
                actual_drift = calc_drift.calc_drift(orig_df, gen_df, columns, verbose=False)
                corr_results = calc_drift.calc_correlation(orig_df, gen_df, verbose=False)
                actual_corr = corr_results.get("pearson", 0.0)
            except Exception as e:
                print(f"  {table}: Error - {e}")
                continue

            drift_diff = abs(actual_drift - target_drift)
            corr_diff = abs(actual_corr - target_corr)

            if table not in all_results:
                all_results[table] = []
            all_results[table].append({
                'base': base,
                'drift': actual_drift,
                'corr': actual_corr,
                'drift_diff': drift_diff,
                'corr_diff': corr_diff
            })

    # Pick best for each table (drift and corr separately)
    best_results = {}
    for table, results in all_results.items():
        if not results:
            continue

        best_drift_entry = min(results, key=lambda x: x['drift_diff'])
        best_corr_entry = min(results, key=lambda x: x['corr_diff'])

        best_results[table] = {
            'best_drift': best_drift_entry['drift'],
            'best_drift_base': best_drift_entry['base'],
            'best_drift_diff': best_drift_entry['drift_diff'],
            'best_corr': best_corr_entry['corr'],
            'best_corr_base': best_corr_entry['base'],
            'best_corr_diff': best_corr_entry['corr_diff'],
        }

    return best_results


def plot_radar(metrics, values, output_filename, yticks, title=None, rmin=None):
    """
    Draw radar chart.

    Args:
        rmin: Minimum radius offset. If set, all values are shifted by rmin to avoid
              small values collapsing to center. The displayed yticks will show original values.
    """
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, projection='polar')

    colors = ['#4C72B0', '#C44E52']
    markers = ['o', 's']

    # Apply rmin offset if specified
    offset = rmin if rmin else 0

    for i, (method, values_method) in enumerate(values.items()):
        values_method = [v + offset for v in values_method]
        values_method = list(values_method)
        values_method += values_method[:1]

        ax.plot(angles, values_method, marker=markers[i], linestyle='-',
                linewidth=6, label=method, color=colors[i], markersize=25)
        ax.fill(angles, values_method, alpha=0.2, color=colors[i])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=55)

    # Adjust yticks for offset
    if offset > 0:
        adjusted_yticks = [t + offset for t in yticks]
        ax.set_yticks(adjusted_yticks)
        ax.set_yticklabels([None] + [f'{t:.2f}' for t in yticks[1:]])
        ax.set_rlim(bottom=offset * 0.35)  # Reduce center whitespace
    else:
        ax.set_yticks(yticks)
        ax.set_yticklabels([None] + yticks[1:])

    ax.tick_params(axis='y', labelsize=55)
    plt.legend(loc='center', bbox_to_anchor=(0.05, 0.95), fontsize=50)
    ax.grid(True)

    if title:
        plt.title(title, fontsize=20, y=1.08)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', format='pdf', dpi=300)
    plt.close()
    print(f"Saved: {output_filename}")


def plot_task(task_name: str, ref_dataset: str, compare_with: str,
              label_real: str, label_gen: str,
              exclude_tables: set, output_dir: str):
    """
    Plot radar charts for a specific task.
    """
    print(f"\n{'='*70}")
    print(f"{task_name}: {compare_with} -> {ref_dataset}")
    print(f"{'='*70}")

    print(f"Getting real drift/corr: {compare_with} -> {ref_dataset}")
    real_values = get_real_drift_corr(compare_with, ref_dataset)

    print(f"Getting generated drift/corr from experiments...")
    gen_values = get_generated_drift_corr(ref_dataset, compare_with)

    # Filter tables
    tables = sorted([t for t in real_values.keys() if t not in exclude_tables])

    print(f"Tables: {tables}")
    print(f"Excluded: {exclude_tables}")

    if not tables:
        print("No tables found!")
        return None

    # Prepare data
    metrics = [t.replace('_', '_\n') for t in tables]

    drift_real = [real_values[t]['drift'] for t in tables]
    drift_gen = [gen_values.get(t, {}).get('best_drift', 0) for t in tables]

    corr_real = [real_values[t]['corr'] for t in tables]
    corr_gen = [gen_values.get(t, {}).get('best_corr', 0) for t in tables]

    # Print summary
    print(f"\nDrift Comparison:")
    print(f"{'Table':<20} {'Real':<12} {'Generated':<12} {'Best Base':<15} {'Diff':<10}")
    print("-" * 70)
    for t in tables:
        real_d = real_values[t]['drift']
        gen_d = gen_values.get(t, {}).get('best_drift', 0)
        base = gen_values.get(t, {}).get('best_drift_base', 'N/A')
        diff = gen_values.get(t, {}).get('best_drift_diff', 0)
        print(f"{t:<20} {real_d:<12.4f} {gen_d:<12.4f} {base:<15} {diff:<10.4f}")

    print(f"\nCorrelation Comparison:")
    print(f"{'Table':<20} {'Real':<12} {'Generated':<12} {'Best Base':<15} {'Diff':<10}")
    print("-" * 70)
    for t in tables:
        real_c = real_values[t]['corr']
        gen_c = gen_values.get(t, {}).get('best_corr', 0)
        base = gen_values.get(t, {}).get('best_corr_base', 'N/A')
        diff = gen_values.get(t, {}).get('best_corr_diff', 0)
        print(f"{t:<20} {real_c:<12.4f} {gen_c:<12.4f} {base:<15} {diff:<10.4f}")

    # Plot drift radar
    values_drift = {
        label_real: drift_real,
        label_gen: drift_gen,
    }
    drift_max = max(max(drift_real), max(drift_gen)) if drift_real and drift_gen else 0.3
    yticks_drift = [round(i * 0.1, 1) for i in range(1, int(drift_max / 0.1) + 2)]

    output_drift = os.path.join(output_dir, f'radar_{task_name.lower()}_drift.pdf')
    plot_radar(metrics, values_drift, output_drift, yticks_drift)

    # Plot correlation radar
    values_corr = {
        label_real: corr_real,
        label_gen: corr_gen,
    }
    corr_max = max(max(corr_real), max(corr_gen)) if corr_real and corr_gen else 0.2
    yticks_corr = [round(i * 0.1, 1) for i in range(0, int(corr_max / 0.1) + 2)]

    output_corr = os.path.join(output_dir, f'radar_{task_name.lower()}_corr.pdf')
    plot_radar(metrics, values_corr, output_corr, yticks_corr)

    return {
        'tables': tables,
        'drift_real': drift_real,
        'drift_gen': drift_gen,
        'corr_real': corr_real,
        'corr_gen': corr_gen,
        'gen_values': gen_values
    }


def get_drift_exp_results(variant_id: int, compare_with: str = "imdb"):
    """
    Get drift and correlation for a specific drift experiment (Task 2).
    Reads from expdir/imdb-{variant_id}/
    """
    exp_dir = os.path.join(EXPDIR, f"imdb-{variant_id}")

    if not os.path.exists(exp_dir):
        print(f"Directory not found: {exp_dir}")
        return {}

    # Load dataset_info for columns
    config_path = f"datasets/{compare_with}/dataset_info.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            dataset_config = json.load(f)
    else:
        dataset_config = {}

    results = {}
    tables = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]

    for table in tables:
        gen_csv = os.path.join(exp_dir, table, f"{table}.drifted.csv")
        orig_csv = f"datasets/{compare_with}/{table}.csv"

        if not os.path.exists(gen_csv) or not os.path.exists(orig_csv):
            continue

        try:
            orig_df = calc_drift.load_csv(orig_csv)
            gen_df = calc_drift.load_csv(gen_csv)
            columns = dataset_config.get(table, {}).get("applicable_columns", [])
            actual_drift = calc_drift.calc_drift(orig_df, gen_df, columns, verbose=False)
            corr_results = calc_drift.calc_correlation(orig_df, gen_df, verbose=False)
            actual_corr = corr_results.get("pearson", 0.0)
        except Exception as e:
            print(f"  {table}: Error - {e}")
            continue

        results[table] = {
            'drift': actual_drift,
            'corr': actual_corr
        }

    return results


def plot_task2(exclude_tables: set, output_dir: str):
    """
    Plot Task 2: Compare drift=0.1, 0.3, 0.5 experiments against imdb.
    """
    print(f"\n{'='*70}")
    print(f"Task 2: Drift Value Experiments (0.1, 0.3, 0.5)")
    print(f"{'='*70}")

    drift_values = [0.1, 0.3, 0.5]
    variant_ids = [1, 3, 5]  # 0.1->1, 0.3->3, 0.5->5

    all_results = {}
    for dv, vid in zip(drift_values, variant_ids):
        print(f"Getting results for drift={dv} (variant_id={vid})...")
        all_results[dv] = get_drift_exp_results(vid, "imdb")

    # Get all tables (union of all experiments)
    all_tables = set()
    for results in all_results.values():
        all_tables.update(results.keys())
    tables = sorted([t for t in all_tables if t not in exclude_tables])

    if not tables:
        print("No tables found!")
        return None

    print(f"Tables: {tables}")
    print(f"Excluded: {exclude_tables}")

    # Print summary
    print(f"\nDrift Comparison:")
    header = f"{'Table':<20}" + "".join([f"{'d='+str(dv):<12}" for dv in drift_values])
    print(header)
    print("-" * (20 + 12 * len(drift_values)))
    for t in tables:
        row = f"{t:<20}"
        for dv in drift_values:
            val = all_results[dv].get(t, {}).get('drift', 0)
            row += f"{val:<12.4f}"
        print(row)

    print(f"\nCorrelation Comparison:")
    print(header)
    print("-" * (20 + 12 * len(drift_values)))
    for t in tables:
        row = f"{t:<20}"
        for dv in drift_values:
            val = all_results[dv].get(t, {}).get('corr', 0)
            row += f"{val:<12.4f}"
        print(row)

    # Prepare data for plotting
    metrics = [t.replace('_', '_\n') for t in tables]

    # Drift radar
    values_drift = {}
    for dv in drift_values:
        values_drift[f'd={dv}'] = [all_results[dv].get(t, {}).get('drift', 0) for t in tables]

    drift_max = max(max(v) for v in values_drift.values()) if values_drift else 0.5
    yticks_drift = [round(i * 0.1, 1) for i in range(1, int(drift_max / 0.1) + 2)]

    output_drift = os.path.join(output_dir, 'radar_task2_drift.pdf')
    plot_radar_multi(metrics, values_drift, output_drift, yticks_drift)

    # Correlation radar
    values_corr = {}
    for dv in drift_values:
        values_corr[f'd={dv}'] = [all_results[dv].get(t, {}).get('corr', 0) for t in tables]

    corr_max = max(max(v) for v in values_corr.values()) if values_corr else 0.2
    yticks_corr = [round(i * 0.1, 1) for i in range(0, int(corr_max / 0.1) + 2)]

    output_corr = os.path.join(output_dir, 'radar_task2_corr.pdf')
    plot_radar_multi(metrics, values_corr, output_corr, yticks_corr)

    return all_results


def plot_radar_multi(metrics, values, output_filename, yticks, title=None, rmin=None):
    """
    Draw radar chart with multiple (>2) series.

    Args:
        rmin: Minimum radius offset for more circular appearance.
    """
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, projection='polar')

    colors = ['#4C72B0', '#C44E52', '#55A868']  # Blue, Red, Green
    markers = ['o', 's', '^']

    offset = rmin if rmin else 0

    for i, (method, values_method) in enumerate(values.items()):
        values_method = [v + offset for v in values_method]
        values_method = list(values_method)
        values_method += values_method[:1]

        ax.plot(angles, values_method, marker=markers[i % len(markers)], linestyle='-',
                linewidth=6, label=method, color=colors[i % len(colors)], markersize=25)
        ax.fill(angles, values_method, alpha=0.15, color=colors[i % len(colors)])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=55)

    # Adjust yticks for offset
    if offset > 0:
        adjusted_yticks = [t + offset for t in yticks]
        ax.set_yticks(adjusted_yticks)
        ax.set_yticklabels([None] + [f'{t:.2f}' for t in yticks[1:]])
        ax.set_rlim(bottom=offset * 0.35)  # Reduce center whitespace
    else:
        ax.set_yticks(yticks)
        ax.set_yticklabels([None] + yticks[1:])

    ax.tick_params(axis='y', labelsize=55)
    plt.legend(loc='center', bbox_to_anchor=(0.05, 0.95), fontsize=50)
    ax.grid(True)

    if title:
        plt.title(title, fontsize=20, y=1.08)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', format='pdf', dpi=300)
    plt.close()
    print(f"Saved: {output_filename}")


def get_generated_drift_corr_filtered(ref_dataset: str, compare_with: str, allowed_bases: set = None):
    """
    Get best generated drift and correlation, optionally filtering by allowed bases.
    """
    config_path = f"datasets/{compare_with}/dataset_info.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            dataset_config = json.load(f)
    else:
        dataset_config = {}

    pattern = f"*_ref_{ref_dataset}"
    exp_dirs = sorted(glob.glob(os.path.join(EXPDIR, pattern)))

    all_results = {}
    real_values = get_real_drift_corr(compare_with, ref_dataset)

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        base = exp_name.replace(f"_ref_{ref_dataset}", "")

        # Filter by allowed bases
        if allowed_bases and base not in allowed_bases:
            continue

        tables = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]

        for table in tables:
            gen_csv = os.path.join(exp_dir, table, f"{table}.drifted.csv")
            if not os.path.exists(gen_csv):
                continue

            target = real_values.get(table, {})
            target_drift = target.get('drift', 0)
            target_corr = target.get('corr', 0)

            orig_csv = f"datasets/{compare_with}/{table}.csv"
            if not os.path.exists(orig_csv):
                continue

            try:
                orig_df = calc_drift.load_csv(orig_csv)
                gen_df = calc_drift.load_csv(gen_csv)
                columns = dataset_config.get(table, {}).get("applicable_columns", [])
                actual_drift = calc_drift.calc_drift(orig_df, gen_df, columns, verbose=False)
                corr_results = calc_drift.calc_correlation(orig_df, gen_df, verbose=False)
                actual_corr = corr_results.get("pearson", 0.0)
            except Exception as e:
                continue

            drift_diff = abs(actual_drift - target_drift)
            corr_diff = abs(actual_corr - target_corr)

            if table not in all_results:
                all_results[table] = []
            all_results[table].append({
                'base': base,
                'drift': actual_drift,
                'corr': actual_corr,
                'drift_diff': drift_diff,
                'corr_diff': corr_diff
            })

    best_results = {}
    for table, results in all_results.items():
        if not results:
            continue
        best_drift_entry = min(results, key=lambda x: x['drift_diff'])
        best_corr_entry = min(results, key=lambda x: x['corr_diff'])
        best_results[table] = {
            'best_drift': best_drift_entry['drift'],
            'best_drift_base': best_drift_entry['base'],
            'best_drift_diff': best_drift_entry['drift_diff'],
            'best_corr': best_corr_entry['corr'],
            'best_corr_base': best_corr_entry['base'],
            'best_corr_diff': best_corr_entry['corr_diff'],
        }

    return best_results


EXPERIMENT_LOGS_DIR = "experiment_logs"


def parse_time_str(time_str: str) -> float:
    """
    Parse time string to seconds.
    Supports formats: "0h 18m 9.6s", "22m 58.1s", "45.3s"
    """
    import re
    total_seconds = 0.0

    # Match hours
    h_match = re.search(r'(\d+)h', time_str)
    if h_match:
        total_seconds += int(h_match.group(1)) * 3600

    # Match minutes
    m_match = re.search(r'(\d+)m', time_str)
    if m_match:
        total_seconds += int(m_match.group(1)) * 60

    # Match seconds
    s_match = re.search(r'([\d.]+)s', time_str)
    if s_match:
        total_seconds += float(s_match.group(1))

    return total_seconds


def load_task4_results(log_dirs: list = None):
    """
    Load Task 4 timing results from experiment logs.
    Scans all log directories and merges results.
    Returns dict: {(dataset, gpu_count): {'train_time': seconds, 'gen_time': seconds}}
    """
    import re

    if log_dirs is None:
        # Scan all log directories, sorted by name (timestamp) so latest comes last
        # Later results will overwrite earlier ones, ensuring we use the latest
        log_dirs = sorted(glob.glob(os.path.join(EXPERIMENT_LOGS_DIR, "*")))
        if not log_dirs:
            print(f"No log directories found in {EXPERIMENT_LOGS_DIR}")
            return {}

    print(f"Loading Task 4 results from {len(log_dirs)} directories (using latest results)")

    results = {}
    pattern = re.compile(r'task4_imdb_to_(imdb_\d+)_(\d+)gpu_(train|gen)\.log')

    for log_dir in log_dirs:
        if not os.path.isdir(log_dir):
            continue

        for log_file in glob.glob(os.path.join(log_dir, "task4_*.log")):
            basename = os.path.basename(log_file)
            match = pattern.match(basename)
            if not match:
                continue

            dataset = match.group(1)
            gpu_count = int(match.group(2))
            phase = match.group(3)

            key = (dataset, gpu_count)
            if key not in results:
                results[key] = {'train_time': None, 'gen_time': None}

            with open(log_file, 'r') as f:
                content = f.read()

            if phase == 'train':
                # Look for "Total time: 0h 18m 9.6s"
                time_match = re.search(r'Total time:\s*([\dh\sm\s.s]+)', content)
                if time_match:
                    results[key]['train_time'] = parse_time_str(time_match.group(1))
            else:  # gen
                # Look for "Generation time (before post-processing): 22m 58.1s"
                # This is the pure generation time without post-processing
                gen_match = re.search(r'Generation time \(before post-processing\):\s*([\dh\sm\s.s]+)', content)
                if gen_match:
                    results[key]['gen_time'] = parse_time_str(gen_match.group(1))
                else:
                    # Fallback to "Total generation time" (includes post-processing)
                    gen_match = re.search(r'Total generation time:\s*([\dh\sm\s.s]+)', content)
                    if gen_match:
                        results[key]['gen_time'] = parse_time_str(gen_match.group(1))
                    else:
                        # Fallback to Total time
                        time_match = re.search(r'Total time:\s*([\dh\sm\s.s]+)', content)
                        if time_match:
                            results[key]['gen_time'] = parse_time_str(time_match.group(1))

    return results


def get_dataset_size(dataset_name: str) -> int:
    """Get total sample count from dataset_info.json."""
    info_path = f"datasets/{dataset_name}/dataset_info.json"
    if not os.path.exists(info_path):
        return 0
    with open(info_path, 'r') as f:
        info = json.load(f)
    total = 0
    for table, data in info.items():
        if data and 'n_samples' in data:
            total += data['n_samples']
    return total


def plot_task4_by_dataset(output_dir: str, gpu_count: int = 4):
    """
    Plot Task 4 bar chart: training and generation time by dataset size.
    X-axis: imdb_2013, imdb_2015, imdb_2017 (using 4 GPUs)
    """
    print(f"\n{'='*70}")
    print(f"Task 4: Time by Dataset (GPU={gpu_count})")
    print(f"{'='*70}")

    results = load_task4_results()

    datasets = ['imdb_2013', 'imdb_2015', 'imdb_2017']
    train_times = []
    gen_times = []
    dataset_sizes = []

    print(f"\n{'Dataset':<15} {'Size':<15} {'Train (min)':<15} {'Gen (min)':<15} {'Total (min)':<15}")
    print("-" * 75)

    for ds in datasets:
        key = (ds, gpu_count)
        train_t = results.get(key, {}).get('train_time')
        gen_t = results.get(key, {}).get('gen_time')

        train_min = train_t / 60 if train_t else 0
        gen_min = gen_t / 60 if gen_t else 0
        size = get_dataset_size(ds)

        train_times.append(train_min)
        gen_times.append(gen_min)
        dataset_sizes.append(size)

        print(f"{ds:<15} {size:<15,} {train_min:<15.2f} {gen_min:<15.2f} {train_min + gen_min:<15.2f}")

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 12))

    x = np.arange(len(datasets))
    width = 0.5

    bars1 = ax.bar(x, train_times, width, label='Training', color='#4C72B0', edgecolor='black', linewidth=2)
    bars2 = ax.bar(x, gen_times, width, bottom=train_times, label='Generation', color='#C44E52', edgecolor='black', linewidth=2)

    # Thicker plot border
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xlabel('', fontsize=60)
    ax.set_ylabel('Time (minutes)', fontsize=60)
    ax.set_xticks(x)
    # X labels with dataset size (in millions)
    xlabels = [f"IMDB-{ds.replace('imdb_', '')[2:]}\n({size/1e6:.1f}M)" for ds, size in zip(datasets, dataset_sizes)]
    ax.set_xticklabels(xlabels, fontsize=60)
    ax.tick_params(axis='y', labelsize=60)
    ax.legend(fontsize=60, loc='upper left')

    # Add value labels
    for i, (t, g) in enumerate(zip(train_times, gen_times)):
        if t > 0:
            ax.text(i, t / 2, f'{t:.1f}', ha='center', va='center', fontsize=60, color='white', fontweight='bold')
        if g > 0:
            ax.text(i, t + g / 2, f'{g:.1f}', ha='center', va='center', fontsize=60, color='white', fontweight='bold')

    # Add margin at top for consistency
    max_height = max(t + g for t, g in zip(train_times, gen_times))
    ax.set_ylim(0, max_height * 1.35)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'bar_task4_by_dataset_{gpu_count}gpu.pdf')
    plt.savefig(output_file, bbox_inches='tight', format='pdf', dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

    return {'datasets': datasets, 'train_times': train_times, 'gen_times': gen_times}


def plot_task4_by_gpu(output_dir: str, dataset: str = 'imdb_2013'):
    """
    Plot Task 4 bar chart: training and generation time by GPU count.
    X-axis: 2, 4, 8 GPUs (using imdb_2013)
    """
    print(f"\n{'='*70}")
    print(f"Task 4: Time by GPU Count (Dataset={dataset})")
    print(f"{'='*70}")

    results = load_task4_results()

    gpu_counts = [2, 4, 8]
    train_times = []
    gen_times = []

    print(f"\n{'GPUs':<10} {'Train (min)':<15} {'Gen (min)':<15} {'Total (min)':<15}")
    print("-" * 55)

    for gpu in gpu_counts:
        key = (dataset, gpu)
        train_t = results.get(key, {}).get('train_time')
        gen_t = results.get(key, {}).get('gen_time')

        train_min = train_t / 60 if train_t else 0
        gen_min = gen_t / 60 if gen_t else 0

        train_times.append(train_min)
        gen_times.append(gen_min)

        print(f"{gpu:<10} {train_min:<15.2f} {gen_min:<15.2f} {train_min + gen_min:<15.2f}")

    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 12))

    x = np.arange(len(gpu_counts))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_times, width, label='Training', color='#4C72B0', edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, gen_times, width, label='Generation', color='#C44E52', edgecolor='black', linewidth=2)

    # Thicker plot border
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xlabel('Number of GPUs', fontsize=60)
    ax.set_ylabel('Time (minutes)', fontsize=60)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gpu_counts], fontsize=60)
    ax.tick_params(axis='y', labelsize=60)
    ax.legend(fontsize=60, loc='upper right')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.3, f'{height:.1f}',
                    ha='center', va='bottom', fontsize=60)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.3, f'{height:.1f}',
                    ha='center', va='bottom', fontsize=60)

    # Add margin at top to prevent value labels from overlapping with border
    max_height = max(max(train_times), max(gen_times))
    ax.set_ylim(0, max_height * 1.15)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'bar_task4_by_gpu_{dataset}.pdf')
    plt.savefig(output_file, bbox_inches='tight', format='pdf', dpi=300)
    plt.close()
    print(f"Saved: {output_file}")

    return {'gpu_counts': gpu_counts, 'train_times': train_times, 'gen_times': gen_times}


def plot_task1_base_comparison(exclude_tables: set, output_dir: str):
    """
    Plot Task 1 comparison: All bases vs without 13/14 vs without 15/16
    """
    print(f"\n{'='*70}")
    print(f"Task 1 Base Comparison: All vs w/o 13-14 vs w/o 15-16")
    print(f"{'='*70}")

    ref_dataset = "imdb_2017"
    compare_with = "imdb"

    # Define base sets
    all_bases = {"imdb", "imdb_2013", "imdb_2014", "imdb_2015", "imdb_2016"}
    no_13_14 = {"imdb", "imdb_2015", "imdb_2016"}
    no_15_16 = {"imdb", "imdb_2013", "imdb_2014"}

    configs = [
        ("All", all_bases),
        ("w/o 13-14", no_13_14),
        ("w/o 15-16", no_15_16),
    ]

    real_values = get_real_drift_corr(compare_with, ref_dataset)
    tables = sorted([t for t in real_values.keys() if t not in exclude_tables])

    if not tables:
        print("No tables found!")
        return None

    print(f"Tables: {tables}")

    all_results = {}
    for label, bases in configs:
        print(f"Getting results for {label} (bases: {bases})...")
        all_results[label] = get_generated_drift_corr_filtered(ref_dataset, compare_with, bases)

    # Print summary
    print(f"\nDrift Comparison:")
    header = f"{'Table':<20}" + "".join([f"{label:<15}" for label, _ in configs])
    print(header)
    print("-" * (20 + 15 * len(configs)))
    for t in tables:
        row = f"{t:<20}"
        for label, _ in configs:
            val = all_results[label].get(t, {}).get('best_drift', 0)
            row += f"{val:<15.4f}"
        print(row)

    print(f"\nCorrelation Comparison:")
    print(header)
    print("-" * (20 + 15 * len(configs)))
    for t in tables:
        row = f"{t:<20}"
        for label, _ in configs:
            val = all_results[label].get(t, {}).get('best_corr', 0)
            row += f"{val:<15.4f}"
        print(row)

    # Prepare data for plotting
    metrics = [t.replace('_', '_\n') for t in tables]

    # Drift radar
    values_drift = {}
    for label, _ in configs:
        values_drift[label] = [all_results[label].get(t, {}).get('best_drift', 0) for t in tables]

    drift_max = max(max(v) for v in values_drift.values()) if values_drift else 0.3
    yticks_drift = [round(i * 0.1, 1) for i in range(1, int(drift_max / 0.1) + 2)]

    output_drift = os.path.join(output_dir, 'radar_task1_base_cmp_drift.pdf')
    plot_radar_multi(metrics, values_drift, output_drift, yticks_drift)

    # Correlation radar
    values_corr = {}
    for label, _ in configs:
        values_corr[label] = [all_results[label].get(t, {}).get('best_corr', 0) for t in tables]

    corr_max = max(max(v) for v in values_corr.values()) if values_corr else 0.2
    yticks_corr = [round(i * 0.1, 1) for i in range(0, int(corr_max / 0.1) + 2)]

    output_corr = os.path.join(output_dir, 'radar_task1_base_cmp_corr.pdf')
    plot_radar_multi(metrics, values_corr, output_corr, yticks_corr)

    return all_results


def plot_task_cached(cache, task_name, real_key, gen_key, label_real, label_gen, exclude_tables, output_dir):
    """Plot task using cached data."""
    print(f"\n{'='*70}")
    print(f"Plotting {task_name}: {label_real} vs {label_gen} (from cache)")
    print(f"{'='*70}")

    real_values = cache.get(real_key, {})
    gen_values = cache.get(gen_key, {})

    tables = sorted([t for t in real_values.keys() if t not in exclude_tables])
    if not tables:
        print("No tables found!")
        return None

    metrics = [t.replace('_', '_\n') for t in tables]
    drift_real = [real_values[t]['drift'] for t in tables]
    drift_gen = [gen_values.get(t, {}).get('best_drift', 0) for t in tables]
    corr_real = [real_values[t]['corr'] for t in tables]
    corr_gen = [gen_values.get(t, {}).get('best_corr', 0) for t in tables]

    # Print summary
    print(f"\nDrift Comparison:")
    print(f"{'Table':<20} {'Real':<12} {'Generated':<12}")
    print("-" * 44)
    for i, t in enumerate(tables):
        print(f"{t:<20} {drift_real[i]:<12.4f} {drift_gen[i]:<12.4f}")

    # Plot drift radar with rmin offset to make chart more circular
    values_drift = {label_real: drift_real, label_gen: drift_gen}
    rmin_drift = 0.25
    yticks_drift = [0, 0.1, 0.2, 0.3]
    output_drift = os.path.join(output_dir, f'radar_{task_name.lower()}_drift.pdf')
    plot_radar(metrics, values_drift, output_drift, yticks_drift, rmin=rmin_drift)

    # Plot correlation radar with rmin offset to make chart more circular
    values_corr = {label_real: corr_real, label_gen: corr_gen}
    rmin_corr = 0.2
    yticks_corr = [0, 0.1, 0.2, 0.3]
    output_corr = os.path.join(output_dir, f'radar_{task_name.lower()}_corr.pdf')
    plot_radar(metrics, values_corr, output_corr, yticks_corr, rmin=rmin_corr)


def plot_task2_cached(cache, exclude_tables, output_dir):
    """Plot Task 2 using cached data."""
    print(f"\n{'='*70}")
    print(f"Plotting Task 2: Drift Value Experiments (from cache)")
    print(f"{'='*70}")

    drift_exp = cache.get('drift_exp', {})
    drift_values = [0.1, 0.3, 0.5]

    all_tables = set()
    for dv in drift_values:
        all_tables.update(drift_exp.get(str(dv), {}).keys())
    tables = sorted([t for t in all_tables if t not in exclude_tables])

    if not tables:
        print("No tables found!")
        return None

    metrics = [t.replace('_', '_\n') for t in tables]

    # Drift radar
    values_drift = {}
    for dv in drift_values:
        values_drift[f'd={dv}'] = [drift_exp.get(str(dv), {}).get(t, {}).get('drift', 0) for t in tables]

    drift_max = max(max(v) for v in values_drift.values()) if values_drift else 0.5
    yticks_drift = [round(i * 0.1, 1) for i in range(1, int(drift_max / 0.1) + 2)]
    output_drift = os.path.join(output_dir, 'radar_task2_drift.pdf')
    plot_radar_multi(metrics, values_drift, output_drift, yticks_drift)

    # Correlation radar with rmin for better appearance
    values_corr = {}
    for dv in drift_values:
        values_corr[f'd={dv}'] = [drift_exp.get(str(dv), {}).get(t, {}).get('corr', 0) for t in tables]

    rmin = 0.15
    yticks_corr = [0, 0.1, 0.2]
    output_corr = os.path.join(output_dir, 'radar_task2_corr.pdf')
    plot_radar_multi(metrics, values_corr, output_corr, yticks_corr, rmin=rmin)


def plot_task1_cmp_cached(cache, exclude_tables, output_dir):
    """Plot Task 1 base comparison: IMDB-17 vs IMDB-17* vs IMDB-17*(w/o 13-14)."""
    print(f"\n{'='*70}")
    print(f"Plotting Task 1 Base Comparison (from cache)")
    print(f"{'='*70}")

    real_values = cache.get('real_imdb_2017', {})
    gen_filtered = cache.get('gen_filtered', {})

    tables = sorted([t for t in real_values.keys() if t not in exclude_tables])
    if not tables:
        print("No tables found!")
        return None

    metrics = [t.replace('_', '_\n') for t in tables]

    # Real values
    drift_real = [real_values[t]['drift'] for t in tables]
    corr_real = [real_values[t]['corr'] for t in tables]

    # Drift radar: Real vs Gen vs Gen*
    values_drift = {
        "Real": drift_real,
        "Gen": [gen_filtered.get("all", {}).get(t, {}).get('best_drift', 0) for t in tables],
        "Gen*": [gen_filtered.get("no_13_14", {}).get(t, {}).get('best_drift', 0) for t in tables],
    }

    drift_max = max(max(v) for v in values_drift.values()) if values_drift else 0.3
    yticks_drift = [round(i * 0.1, 1) for i in range(1, int(drift_max / 0.1) + 2)]
    output_drift = os.path.join(output_dir, 'radar_task1_base_cmp_drift.pdf')
    plot_radar_multi(metrics, values_drift, output_drift, yticks_drift)

    # Correlation radar: Real vs Gen vs Gen*
    values_corr = {
        "Real": corr_real,
        "Gen": [gen_filtered.get("all", {}).get(t, {}).get('best_corr', 0) for t in tables],
        "Gen*": [gen_filtered.get("no_13_14", {}).get(t, {}).get('best_corr', 0) for t in tables],
    }

    rmin = 0.2
    yticks_corr = [0, 0.1, 0.2, 0.3]
    output_corr = os.path.join(output_dir, 'radar_task1_base_cmp_corr.pdf')
    plot_radar_multi(metrics, values_corr, output_corr, yticks_corr, rmin=rmin)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot data generation quality charts")
    parser.add_argument("task", choices=["compute", "task1", "task2", "task3", "task1-cmp", "task4-data", "task4-gpu", "all"],
                        help="compute: pre-compute all data; task1-4: plot specific task; all: plot all")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for PDFs")
    parser.add_argument("--exclude", type=str, nargs="*", default=["title", "aka_name"],
                        help="Tables to exclude")
    parser.add_argument("--gpu", type=int, default=4, help="GPU count for task4-data (default: 4)")
    parser.add_argument("--dataset", type=str, default="imdb_2013", help="Dataset for task4-gpu (default: imdb_2013)")
    parser.add_argument("--update-cache", action="store_true", help="Force recompute and update cache")
    args = parser.parse_args()

    exclude_tables = set(args.exclude) if args.exclude else set()
    os.makedirs(args.output_dir, exist_ok=True)

    # Handle compute command
    if args.task == "compute":
        cache = compute_all_data()
        save_cache(cache)
        return

    # For plotting tasks, load or compute cache
    cache = load_cache()
    if args.update_cache or not cache:
        if not cache:
            print("No cache found, computing...")
        else:
            print("Updating cache...")
        cache = compute_all_data()
        save_cache(cache)

    # Plot tasks using cached data
    if args.task in ["task1", "all"]:
        plot_task_cached(cache, "task1", "real_imdb_2017", "gen_imdb_2017",
                         "Real", "Gen", exclude_tables, args.output_dir)

    if args.task in ["task2", "all"]:
        plot_task2_cached(cache, exclude_tables, args.output_dir)

    if args.task in ["task3", "all"]:
        plot_task_cached(cache, "task3", "real_imdb_2015", "gen_imdb_2015",
                         "Real", "Gen", exclude_tables, args.output_dir)

    if args.task in ["task1-cmp", "all"]:
        plot_task1_cmp_cached(cache, exclude_tables, args.output_dir)

    if args.task in ["task4-data", "all"]:
        plot_task4_by_dataset(args.output_dir, gpu_count=args.gpu)

    if args.task in ["task4-gpu", "all"]:
        plot_task4_by_gpu(args.output_dir, dataset=args.dataset)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
