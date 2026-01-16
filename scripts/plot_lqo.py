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
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['legend.fontsize'] = 10

# Colors matching the gnuplot script (lt 1-10)
COLORS = {
    'PostgreSQL': '#7f7f7f',  # lt 8 - middle gray
    'Bao': '#1f77b4',         # lt 1 - blue
    'Balsa': '#ff7f0e',       # lt 2 - orange
    'HybridQO': '#2ca02c',    # lt 3 - green
    'Lero': '#d62728',        # lt 4 - red
}

# Hatch patterns - only PostgreSQL has hatch, others are solid
HATCHES = {
    'PostgreSQL': '//',   # pattern 2
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
    systems = ['PostgreSQL', 'Bao', 'Balsa', 'HybridQO', 'Lero']
    n_groups = len(data[systems[0]])
    n_systems = len(systems)

    # Bar width and positions
    bar_width = 0.15
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

    # Set x-axis range
    ax.set_xlim(-0.8, n_groups - 0.2)

    # Legend - split into 2 rows (3+2) if 5 systems
    if show_legend:
        ncol = 3 if n_systems == 5 else n_systems
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22),
                  ncol=ncol, frameon=False)

    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
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
        'HybridQO': [130.904857, 227.487, 261.438785, 341.4895388],
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
        figsize=(5.3, 3.5),
        y_tick_interval=50
    )


def plot_stack_performance(output_dir='plots'):
    """
    Plot the STACK dataset performance comparison.
    Data from drift-real-fixed-train-stack.csv
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Data: STACK-08, STACK-10, STACK-10*
    # Columns in CSV: name, PostgreSQL, HybridQO, Bao, Balsa, Lero
    data = {
        'PostgreSQL': [3193.07, 4858.509, 4976.129],
        'HybridQO': [3381.894, 4562.481, 4612.347],
        'Bao': [2851.474, 4836.916, 4883.314],
        'Balsa': [2713.415, 5321.234, 5468.813],
        'Lero': [3213.145, 4773.412, 4741.21],
    }

    x_labels = ['STACK-08', 'STACK-10', 'STACK-10*']

    # Plot with legend (consistent with demo style)
    output_file = os.path.join(output_dir, 'qo_data_drift_real_fix_train_w_stack.pdf')
    plot_drift_performance(
        data,
        output_file,
        x_labels=x_labels,
        xlabel="Testing Data",
        ylabel="Execution Time (ms)",
        show_legend=True,
        figsize=(5.3, 3.5),
        y_tick_interval=1000
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
        'HybridQO': [130.904857, 231.438785, 237.487],
        'Bao': [110.792436, 216.278996, 223.234],
        'Balsa': [110.639, 265.181, 262.764],
        'Lero': [132.626952, 221.334736, 220.659],
    }

    x_labels = ['IMDB-13', 'IMDB-17', 'IMDB-17*']

    # Plot with legend (consistent with other plots)
    output_file = os.path.join(output_dir, 'qo_data_drift_real_fix_train_w_13.pdf')
    plot_drift_performance(
        data,
        output_file,
        x_labels=x_labels,
        xlabel="Testing Data",
        ylabel="Execution Time (s)",
        show_legend=True,
        figsize=(5.3, 3.5),
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
        system_names = ['PostgreSQL', 'Bao', 'Balsa', 'HybridQO', 'Lero']

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
