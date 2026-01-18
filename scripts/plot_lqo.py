#!/usr/bin/env python3
"""
LQO Performance Plotting Script
Generates bar charts comparing query optimizer performance under data drift.
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Set up matplotlib to use Helvetica font (same style as plot_data_gen.py)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 64
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 64
plt.rcParams['xtick.labelsize'] = 64
plt.rcParams['ytick.labelsize'] = 64
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['legend.fontsize'] = 64

# Colors matching the gnuplot script (lt 1-10)
COLORS = {
    'PostgreSQL': '#d0d0d0',  # light gray
    'Bao': '#1f77b4',         # lt 1 - blue
    'Balsa': '#ff7f0e',       # lt 2 - orange
    'HybridQO': '#2ca02c',    # lt 3 - green
    'Lero': '#2ca02c',        # lt 3 - green (was HybridQO's color)
}

# Hatch patterns - only PostgreSQL has hatch, others are solid
HATCHES = {
    'PostgreSQL': '',     # solid (no pattern)
    'Bao': '',            # solid
    'Balsa': '',          # solid
    'HybridQO': '',       # solid
    'Lero': '',           # solid
}


def plot_drift_performance(data, output_file, x_labels=None, xlabel="Drift Factor",
                           ylabel="Execution Time (s)", title=None, show_legend=True,
                           figsize=(5.3, 3), y_tick_interval=50):
    """
    Plot clustered bar chart for LQO performance comparison.

    Args:
        data: dict with keys as system names and values as lists of execution times
        output_file: output PDF file path
        x_labels: labels for x-axis groups
        xlabel: x-axis label
        ylabel: y-axis label
        title: plot title (optional)
        show_legend: whether to show legend
        figsize: figure size in inches
        y_tick_interval: interval for y-axis ticks
    """
    systems = ['PostgreSQL', 'Bao', 'Balsa', 'Lero']  # 'HybridQO' temporarily commented out
    n_groups = len(data[systems[0]])
    n_systems = len(systems)

    # Bar width and positions
    bar_width = 0.18
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars for each system
    for i, system in enumerate(systems):
        offset = (i - n_systems / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, data[system], bar_width,
                      label=system,
                      color=COLORS[system],
                      hatch=HATCHES[system],
                      edgecolor='black',
                      linewidth=0.5)

    # Customize axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if x_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

    if title:
        ax.set_title(title)

    # Set y-axis ticks
    y_max = max(max(data[s]) for s in systems)
    y_ticks = np.arange(0, y_max + y_tick_interval, y_tick_interval)
    ax.set_yticks(y_ticks)

    # Set x-axis range (balanced margins)
    ax.set_xlim(-0.65, n_groups - 0.35)

    # Legend - place above chart, centered in canvas
    if show_legend:
        ax.legend(loc='lower center', bbox_to_anchor=(0.43, 1.02), ncol=2,
                  fontsize=64, frameon=True, facecolor='white', edgecolor='none',
                  columnspacing=0.5, handletextpad=0.3)

    plt.tight_layout(pad=0)
    plt.savefig(output_file, format='pdf', bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close()
    print(f"Saved plot to {output_file}")


def plot_drift_real_fixed_train(output_dir='plots'):
    """
    Plot the drift performance with fixed training data.
    Replicates the gnuplot script output.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Data from drift-real-fixed-train.csv
    # Columns: drift_factor, PostgreSQL, HybridQO, Bao, Balsa, Lero
    data = {
        'PostgreSQL': [134.436911, 213.112, 237.218039, 308.3834507],
        # 'HybridQO': [130.904857, 227.487, 261.438785, 341.4895388],
        'Bao': [110.792436, 213.234, 256.278996, 353.3771498],
        'Balsa': [110.639, 252.764, 275.181, 461.69019],
        'Lero': [132.626952, 210.659, 251.334736, 325.552354],
    }

    x_labels = ['0', '0.1', '0.3', '0.5']

    # Plot with legend
    output_file = os.path.join(output_dir, 'qo_data_drift_real_fix_train.pdf')
    plot_drift_performance(
        data,
        output_file,
        x_labels=x_labels,
        xlabel="Drift Factor",
        ylabel="Execution Time (s)",
        show_legend=True,
        figsize=(16, 12),
        y_tick_interval=50
    )


def plot_stack_performance(output_dir='plots'):
    """
    Plot the STACK dataset performance comparison.
    Data from drift-real-fixed-train-stack.csv
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Data: STACK-08, STACK-10, STACK-10* (converted to seconds)
    # Columns in CSV: name, PostgreSQL, HybridQO, Bao, Balsa, Lero
    data = {
        'PostgreSQL': [3.19307, 4.858509, 4.976129],
        # 'HybridQO': [3.381894, 4.562481, 4.612347],
        'Bao': [2.851474, 4.836916, 4.883314],
        'Balsa': [2.713415, 5.321234, 5.468813],
        'Lero': [3.213145, 4.773412, 4.74121],
    }

    x_labels = ['2008', '2010', '2010*']

    # Plot with legend
    output_file = os.path.join(output_dir, 'qo_data_drift_real_fix_train_w_stack.pdf')
    plot_drift_performance(
        data,
        output_file,
        x_labels=x_labels,
        xlabel="STACK",
        ylabel="Execution Time (s)",
        show_legend=True,
        figsize=(14, 12),
        y_tick_interval=1
    )


def plot_imdb_performance(output_dir='plots'):
    """
    Plot the IMDB dataset performance comparison.
    Data from drift-real-fixed-train.csv (IMDB-13, IMDB-17, IMDB-17*)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Data: IMDB-13, IMDB-17, IMDB-17*
    # Columns in CSV: name, PostgreSQL, HybridQO, Bao, Balsa, Lero
    data = {
        'PostgreSQL': [134.436911, 227.218039, 223.112],
        # 'HybridQO': [130.904857, 231.438785, 237.487],
        'Bao': [110.792436, 216.278996, 223.234],
        'Balsa': [110.639, 265.181, 262.764],
        'Lero': [132.626952, 221.334736, 220.659],
    }

    x_labels = ['2013', '2017', '2017*']

    # Plot with legend
    output_file = os.path.join(output_dir, 'qo_data_drift_real_fix_train_w_13.pdf')
    plot_drift_performance(
        data,
        output_file,
        x_labels=x_labels,
        xlabel="IMDB",
        ylabel="Execution Time (s)",
        show_legend=True,
        figsize=(14, 12),
        y_tick_interval=50
    )


def plot_from_csv(csv_file, output_file, x_col=0, y_cols=None, system_names=None,
                  x_labels=None, xlabel="Drift Factor", ylabel="Execution Time (s)",
                  show_legend=True, figsize=(5.3, 3), y_tick_interval=50):
    """
    Plot from a CSV file.

    Args:
        csv_file: path to CSV file
        output_file: output PDF file path
        x_col: column index for x-axis values (used as labels if x_labels not provided)
        y_cols: list of column indices for y values (default: [1,2,3,4,5])
        system_names: list of system names corresponding to y_cols
        x_labels: custom x-axis labels
        xlabel: x-axis label
        ylabel: y-axis label
        show_legend: whether to show legend
        figsize: figure size
        y_tick_interval: interval for y-axis ticks
    """
    import csv

    if y_cols is None:
        y_cols = [1, 2, 3, 4, 5]

    if system_names is None:
        system_names = ['PostgreSQL', 'Bao', 'Balsa', 'Lero']  # 'HybridQO' temporarily commented out

    # Read CSV
    data = {name: [] for name in system_names}
    x_values = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            try:
                x_values.append(row[x_col])
                for i, col in enumerate(y_cols):
                    data[system_names[i]].append(float(row[col]))
            except (ValueError, IndexError):
                continue

    if x_labels is None:
        x_labels = x_values

    plot_drift_performance(
        data, output_file, x_labels=x_labels,
        xlabel=xlabel, ylabel=ylabel,
        show_legend=show_legend, figsize=figsize,
        y_tick_interval=y_tick_interval
    )


def main():
    parser = argparse.ArgumentParser(description='Plot LQO performance comparison')
    parser.add_argument('--csv', type=str, help='Input CSV file')
    parser.add_argument('--output', '-o', type=str, default='lqo_performance.pdf',
                        help='Output PDF file')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Output directory for plots')
    parser.add_argument('--demo', action='store_true',
                        help='Generate demo plot with hardcoded data')
    parser.add_argument('--stack', action='store_true',
                        help='Generate STACK dataset plot')
    parser.add_argument('--imdb', action='store_true',
                        help='Generate IMDB dataset plot')
    parser.add_argument('--all', action='store_true',
                        help='Generate all plots (demo + stack + imdb)')
    parser.add_argument('--xlabel', type=str, default='Drift Factor',
                        help='X-axis label')
    parser.add_argument('--ylabel', type=str, default='Execution Time (s)',
                        help='Y-axis label')
    parser.add_argument('--no-legend', action='store_true',
                        help='Hide legend')
    parser.add_argument('--figwidth', type=float, default=5.3,
                        help='Figure width in inches')
    parser.add_argument('--figheight', type=float, default=3.0,
                        help='Figure height in inches')
    parser.add_argument('--ytick', type=float, default=50,
                        help='Y-axis tick interval')

    args = parser.parse_args()

    if args.all:
        print("Generating all plots...")
        plot_drift_real_fixed_train(args.output_dir)
        plot_stack_performance(args.output_dir)
        plot_imdb_performance(args.output_dir)
    elif args.stack:
        print("Generating STACK plots...")
        plot_stack_performance(args.output_dir)
    elif args.imdb:
        print("Generating IMDB plots...")
        plot_imdb_performance(args.output_dir)
    elif args.demo:
        print("Generating demo plots...")
        plot_drift_real_fixed_train(args.output_dir)
    elif args.csv:
        plot_from_csv(
            args.csv, args.output,
            xlabel=args.xlabel, ylabel=args.ylabel,
            show_legend=not args.no_legend,
            figsize=(args.figwidth, args.figheight),
            y_tick_interval=args.ytick
        )
    else:
        # Default: generate demo plot
        print("No input specified. Generating demo plots...")
        print("Usage: python plot_lqo.py --csv data.csv -o output.pdf")
        print("       python plot_lqo.py --demo")
        print("       python plot_lqo.py --stack")
        print("       python plot_lqo.py --imdb")
        print("       python plot_lqo.py --all")
        plot_drift_real_fixed_train(args.output_dir)


if __name__ == '__main__':
    main()
