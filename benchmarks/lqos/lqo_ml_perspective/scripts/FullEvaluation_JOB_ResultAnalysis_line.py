#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

# In[2]:


import os
import sys

if os.getcwd().split('/')[-1] == 'notebooks':
    os.chdir('../')
os.getcwd()

# In[3]:


import os
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import LinearSegmentedColormap, to_rgba_array
from matplotlib.ticker import LogLocator

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Filter the DataFrame for specific experiments
relevant_experiments = [
    'leave_one_out_split_1', 'leave_one_out_split_2', 'leave_one_out_split_3',
    'random_split_1', 'random_split_2', 'random_split_3',
    'base_query_split_1', 'base_query_split_2', 'base_query_split_3'
]


def draw_query_by_query_multiple_lines_comapre_with_me(_df_agg):
    used_query = ['01a', '01b', '01c', '01d', '02a', '02b', '02c', '02d', '03a', '03b', '03c', '04a', '04b', '04c',
                  '05a',
                  '05b', '05c',
                  '06a', '06b', '06c', '06d', '06e', '06f', '07a', '07b', '07c', '08a', '08b', '08c', '08d', '09a',
                  '09b',
                  '09c', '09d',
                  '10a', '10b', '10c', '11a', '11b', '11c', '11d', '12a', '12b', '12c', '13a', '13b', '13c', '13d',
                  '14a',
                  '14b', '14c',
                  '15a', '15b', '15c', '15d', '16a', '16b', '16c', '16d', '17a', '17b', '17c', '17d', '17e', '17f',
                  '18a',
                  '18b', '18c',
                  '19a', '19b', '19c', '19d', '20a', '20b', '20c', '21a', '21b', '21c', '22a', '22b', '22c', '22d',
                  '23a',
                  '23b', '23c',
                  '24a', '24b', '25a', '25b', '25c', '26a', '26b', '26c', '27a', '27b', '27c', '28a', '28b', '28c',
                  '29a',
                  '29b', '29c',
                  '30a', '30b', '30c', '31a', '31b', '31c', '32a', '32b', '33a', '33b', '33c']
    used_query = [f"{int(q[:-1])}{q[-1]}.sql" for q in used_query]

    # Load the existing CSV file
    csv_file_path = '/Users/kevin/project_python/AI4QueryOptimizer/AI4QueryOptimizer/baseline/qp_evaluation/experiments/result/custom_optimization_results.csv'
    my_res = pd.read_csv(csv_file_path)
    print(my_res.head())

    # Filter the DataFrame to only include rows where 'query_id' is in used_query
    _df_my_method = my_res[my_res['query_id'].isin(used_query)]
    _df_my_method['total_time'] = _df_my_method['inference_time'] + _df_my_method['execution_time']

    # First, ensure the 'query_ident' in _df_agg matches the format used in _df_my_method's 'query_id'
    _df_agg['query_id'] = _df_agg['query_ident'].apply(lambda x: f"{int(x[:-1])}{x[-1]}.sql")

    # Merge the execution and total time from _df_my_method into _df_agg
    _df_agg = _df_agg.merge(_df_my_method[['query_id', 'execution_time', 'total_time']],
                            on='query_id',
                            how='left',
                            suffixes=('', '_my_method'))

    # Rename merged columns for clarity
    _df_agg.rename(columns={'execution_time_my_method': 'my_method_execution_time',
                            'total_time_my_method': 'my_method_total_time'}, inplace=True)

    _df_agg = _df_agg[_df_agg['experiment'].isin(relevant_experiments) & (_df_agg['split'] == 'test')]

    # Remove rows where 'my_method_execution_time' or 'my_method_total_time' are NaN
    _df_agg.dropna(subset=['my_method_execution_time', 'my_method_total_time'], inplace=True)

    # Convert necessary columns to numeric if not already
    numeric_cols = ['execution_time', 'total_time']
    _df_agg[numeric_cols] = _df_agg[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Create a unique identifier for each combination of experiment and query
    _df_agg['query_experiment'] = _df_agg['experiment'] + '____' + _df_agg['query_ident']

    # Sort by this new column to maintain a consistent order
    _df_agg['experiment'] = pd.Categorical(_df_agg['experiment'], categories=relevant_experiments, ordered=True)
    _df_agg = _df_agg.sort_values(by=['experiment', 'query_ident'])

    # Assign a numeric index based on the unique 'query_experiment' to serve as the x-axis
    unique_queries = _df_agg['query_experiment'].unique()
    query_index = {query: idx for idx, query in enumerate(unique_queries)}
    _df_agg['query_index'] = _df_agg['query_experiment'].apply(lambda x: query_index[x])

    unique_combinations_execute_time_my_mthod = _df_agg[['query_index', 'my_method_execution_time']].drop_duplicates()
    unique_combinations_execute_time_total_time = _df_agg[['query_index', 'my_method_total_time']].drop_duplicates()

    # Identify all methods excluding PostgreSQL
    methods = _df_agg['method'].unique()
    methods = [m for m in methods if m != 'PostgreSQL']

    # Determine number of rows for subplots
    num_rows = len(methods)
    fig, axes = plt.subplots(num_rows, 1, figsize=(15, 2 * num_rows))

    # Ensure 'axes' is iterable
    if num_rows == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        # Pivot the DataFrame to get each method as columns with their respective times
        pivot_df = _df_agg[_df_agg['method'].isin([method, 'PostgreSQL'])]
        execution_time_df = pivot_df.pivot_table(index='query_index', columns='method', values='execution_time',
                                                 aggfunc='mean')
        total_time_df = pivot_df.pivot_table(index='query_index', columns='method', values='total_time',
                                             aggfunc='mean')

        # Plot PostgreSQL and the current method for execution time
        if 'PostgreSQL' in execution_time_df.columns:
            ax.plot(execution_time_df.index, total_time_df['PostgreSQL'], label="PostgreSQL (Total Time)",
                    color='blue')
        if method in execution_time_df.columns:
            # ax.plot(unique_combinations_execute_time_my_mthod['query_index'],
            #         unique_combinations_execute_time_my_mthod['my_method_execution_time'] * 1000,
            #         label="Our Solution (Execution Time)", color='red')

            ax.plot(unique_combinations_execute_time_total_time['query_index'],
                    unique_combinations_execute_time_total_time['my_method_total_time'] * 1000,
                    label="Our Solution (Total Time)", color='green')

            #
            # ax.plot(execution_time_df.index, execution_time_df[method], label=f"{method} (Execution Time)",
            #         color='red')
            # ax.plot(total_time_df.index, total_time_df[method], label=f"{method} (Total Time)",
            #         color='green')

        # Add vertical lines at transitions between different workloads
        previous_experiment = None
        for idx, query in enumerate(unique_queries):
            current_experiment = query.split('____')[0]
            if previous_experiment and current_experiment != previous_experiment:
                ax.axvline(x=idx - 0.5, color='black', linestyle='--')
            previous_experiment = current_experiment

        ax.set_ylabel('Log of Time (ms)')
        # ax.set_ylim([0, 10000])
        ax.set_yscale("log")  # Set y-axis to logarithmic scale
        ax.legend(ncol=5)
        ax.grid(True)

    plt.tight_layout()  # Adjust layout to make room for subplots
    plt.savefig(f'combined_plot_with_me.pdf', format='pdf')  # Save the entire figure as a PDF
    plt.close()


def draw_query_by_query_bar_comapre_with_me(_df_agg):
    used_query = ['01a', '01b', '01c', '01d', '02a', '02b', '02c', '02d', '03a', '03b', '03c', '04a', '04b', '04c',
                  '05a',
                  '05b', '05c',
                  '06a', '06b', '06c', '06d', '06e', '06f', '07a', '07b', '07c', '08a', '08b', '08c', '08d', '09a',
                  '09b',
                  '09c', '09d',
                  '10a', '10b', '10c', '11a', '11b', '11c', '11d', '12a', '12b', '12c', '13a', '13b', '13c', '13d',
                  '14a',
                  '14b', '14c',
                  '15a', '15b', '15c', '15d', '16a', '16b', '16c', '16d', '17a', '17b', '17c', '17d', '17e', '17f',
                  '18a',
                  '18b', '18c',
                  '19a', '19b', '19c', '19d', '20a', '20b', '20c', '21a', '21b', '21c', '22a', '22b', '22c', '22d',
                  '23a',
                  '23b', '23c',
                  '24a', '24b', '25a', '25b', '25c', '26a', '26b', '26c', '27a', '27b', '27c', '28a', '28b', '28c',
                  '29a',
                  '29b', '29c',
                  '30a', '30b', '30c', '31a', '31b', '31c', '32a', '32b', '33a', '33b', '33c']
    used_query = [f"{int(q[:-1])}{q[-1]}.sql" for q in used_query]

    # Load the existing CSV file
    csv_file_path = '/Users/kevin/project_python/AI4QueryOptimizer/AI4QueryOptimizer/baseline/qp_evaluation/experiments/result/custom_optimization_results.csv'
    my_res = pd.read_csv(csv_file_path)
    print(my_res.head())

    # Filter the DataFrame to only include rows where 'query_id' is in used_query
    _df_my_method = my_res[my_res['query_id'].isin(used_query)]
    _df_my_method['total_time'] = _df_my_method['inference_time'] + _df_my_method['execution_time']

    # First, ensure the 'query_ident' in _df_agg matches the format used in _df_my_method's 'query_id'
    _df_agg['query_id'] = _df_agg['query_ident'].apply(lambda x: f"{int(x[:-1])}{x[-1]}.sql")

    # Merge the execution and total time from _df_my_method into _df_agg
    _df_agg = _df_agg.merge(_df_my_method[['query_id', 'execution_time', 'total_time']],
                            on='query_id',
                            how='left',
                            suffixes=('', '_my_method'))

    # Rename merged columns for clarity
    _df_agg.rename(columns={'execution_time_my_method': 'my_method_execution_time',
                            'total_time_my_method': 'my_method_total_time'}, inplace=True)

    _df_agg = _df_agg[_df_agg['experiment'].isin(relevant_experiments) & (_df_agg['split'] == 'test')]

    # Remove rows where 'my_method_execution_time' or 'my_method_total_time' are NaN
    _df_agg.dropna(subset=['my_method_execution_time', 'my_method_total_time'], inplace=True)

    # Convert necessary columns to numeric if not already
    numeric_cols = ['execution_time', 'total_time']
    _df_agg[numeric_cols] = _df_agg[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Ensure experiments are in a specific order
    _df_agg['experiment'] = pd.Categorical(_df_agg['experiment'], categories=relevant_experiments, ordered=True)
    _df_agg = _df_agg.sort_values(by=['experiment', 'method', 'query_ident'])

    # Identify all methods excluding PostgreSQL
    methods = _df_agg['method'].unique()
    methods = [m for m in methods if m != 'PostgreSQL']

    num_methods = len(methods)
    fig, axes = plt.subplots(num_methods, 1, figsize=(10, 1.5 * num_methods))

    if num_methods == 1:
        axes = [axes]  # Ensure axes is iterable

    for ax, method in zip(axes, methods):
        # Data for the current method and PostgreSQL
        method_data = _df_agg[_df_agg['method'].isin([method, 'PostgreSQL'])]
        total_means = method_data.pivot_table(index='experiment', columns='method', values='total_time', aggfunc='sum')

        our_solution_means = method_data.groupby('experiment')['my_method_total_time'].sum().reset_index()

        # Set up bar positions
        bar_width = 0.5
        indices = np.arange(len(relevant_experiments)) * 3  # Expand indices to space out the bars

        ax.bar(indices, our_solution_means["my_method_total_time"]*1000, bar_width, label=f'Our Solution Total Time', color='orange')
        ax.bar(indices + bar_width, total_means['PostgreSQL'], bar_width, label='PostgreSQL Total Time', color='blue')

        ax.set_ylabel('Time (ms)')
        ax.set_xticks(indices)
        x_labels = ["wl_shift_" + str(i) for i in range(len(relevant_experiments))]
        ax.set_yscale("log")  # Set y-axis to logarithmic scale
        ax.set_xticklabels(x_labels)
        ax.legend(ncol=5)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('method_comparison_bars_with_me.pdf', format='pdf')
    plt.close()


def draw_query_by_query_bar_charts(_df_agg):
    _df_agg = _df_agg[_df_agg['experiment'].isin(relevant_experiments) & (_df_agg['split'] == 'test')]

    # Convert necessary columns to numeric if not already
    numeric_cols = ['execution_time', 'total_time']
    _df_agg[numeric_cols] = _df_agg[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Ensure experiments are in a specific order
    _df_agg['experiment'] = pd.Categorical(_df_agg['experiment'], categories=relevant_experiments, ordered=True)
    _df_agg = _df_agg.sort_values(by=['experiment', 'method', 'query_ident'])

    # Identify all methods excluding PostgreSQL
    methods = _df_agg['method'].unique()
    methods = [m for m in methods if m != 'PostgreSQL']

    num_methods = len(methods)
    fig, axes = plt.subplots(num_methods, 1, figsize=(10, 1.5 * num_methods))

    if num_methods == 1:
        axes = [axes]  # Ensure axes is iterable

    for ax, method in zip(axes, methods):
        # Data for the current method and PostgreSQL
        method_data = _df_agg[_df_agg['method'].isin([method, 'PostgreSQL'])]

        # Mean values for each experiment
        execution_means = method_data.pivot_table(index='experiment', columns='method', values='execution_time',
                                                  aggfunc='sum')
        total_means = method_data.pivot_table(index='experiment', columns='method', values='total_time',
                                              aggfunc='sum')

        # Set up bar positions
        bar_width = 0.5
        indices = np.arange(len(relevant_experiments)) * 3  # Expand indices to space out the bars

        # Execution time bars for method
        ax.bar(indices - bar_width, execution_means[method], bar_width, label=f'{method} Execution Time', color='red')

        # Total time bars for method
        ax.bar(indices, total_means[method], bar_width, label=f'{method} Total Time', color='orange')

        # Total time bars for PostgreSQL
        ax.bar(indices + bar_width, total_means['PostgreSQL'], bar_width, label='PostgreSQL Total Time', color='blue')

        ax.set_ylabel('Time (ms)')
        ax.set_xticks(indices)
        x_labels = ["wl_shift_" + str(i) for i in range(len(relevant_experiments))]
        ax.set_yscale("log")  # Set y-axis to logarithmic scale
        ax.set_xticklabels(x_labels)
        ax.legend(ncol=5)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('method_comparison_bars.pdf', format='pdf')
    plt.close()


def draw_query_by_query_multiple_lines(_df_agg):
    _df_agg = _df_agg[_df_agg['experiment'].isin(relevant_experiments) & (_df_agg['split'] == 'test')]

    # Convert necessary columns to numeric if not already
    numeric_cols = ['execution_time', 'total_time']
    _df_agg[numeric_cols] = _df_agg[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Create a unique identifier for each combination of experiment and query
    _df_agg['query_experiment'] = _df_agg['experiment'] + '____' + _df_agg['query_ident']

    # Sort by this new column to maintain a consistent order
    _df_agg['experiment'] = pd.Categorical(_df_agg['experiment'], categories=relevant_experiments, ordered=True)
    _df_agg = _df_agg.sort_values(by=['experiment', 'query_ident'])

    # Assign a numeric index based on the unique 'query_experiment' to serve as the x-axis
    unique_queries = _df_agg['query_experiment'].unique()
    query_index = {query: idx for idx, query in enumerate(unique_queries)}
    _df_agg['query_index'] = _df_agg['query_experiment'].apply(lambda x: query_index[x])

    # Identify all methods excluding PostgreSQL
    methods = _df_agg['method'].unique()
    methods = [m for m in methods if m != 'PostgreSQL']

    # Determine number of rows for subplots
    num_rows = len(methods)
    fig, axes = plt.subplots(num_rows, 1, figsize=(15, 2 * num_rows))

    # Ensure 'axes' is iterable
    if num_rows == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        # Pivot the DataFrame to get each method as columns with their respective times
        pivot_df = _df_agg[_df_agg['method'].isin([method, 'PostgreSQL'])]
        execution_time_df = pivot_df.pivot_table(index='query_index', columns='method', values='execution_time',
                                                 aggfunc='mean')
        total_time_df = pivot_df.pivot_table(index='query_index', columns='method', values='total_time',
                                             aggfunc='mean')

        # Plot PostgreSQL and the current method for execution time
        if 'PostgreSQL' in execution_time_df.columns:
            ax.plot(execution_time_df.index, total_time_df['PostgreSQL'], label="PostgreSQL (Total Time)",
                    color='blue')
        if method in execution_time_df.columns:
            ax.plot(execution_time_df.index, execution_time_df[method], label=f"{method} (Execution Time)",
                    color='red')
            ax.plot(total_time_df.index, total_time_df[method], label=f"{method} (Total Time)",
                    color='green')  # Plotting total time

        # Add vertical lines at transitions between different workloads
        previous_experiment = None
        for idx, query in enumerate(unique_queries):
            current_experiment = query.split('____')[0]
            if previous_experiment and current_experiment != previous_experiment:
                ax.axvline(x=idx - 0.5, color='black', linestyle='--')
            previous_experiment = current_experiment

        ax.set_ylabel('Time (ms)')
        ax.set_yscale("log")  # Set y-axis to logarithmic scale
        ax.legend(ncol=5)
        ax.grid(True)

    plt.tight_layout()  # Adjust layout to make room for subplots
    plt.savefig(f'combined_plot.pdf', format='pdf')  # Save the entire figure as a PDF
    plt.close()


def draw_query_by_query(_df_agg):
    # Filter the DataFrame for specific experiments

    _df_agg = _df_agg[_df_agg['experiment'].isin(relevant_experiments) & (_df_agg['split'] == 'test')]

    # Convert necessary columns to numeric if not already
    time_measurement = 'execution_time'
    # time_measurement = 'total_time'
    numeric_cols = [time_measurement]

    _df_agg[numeric_cols] = _df_agg[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Create a unique identifier for each combination of experiment and query
    _df_agg['query_experiment'] = _df_agg['experiment'] + '____' + _df_agg['query_ident']

    # Sort by this new column to maintain a consistent order
    _df_agg['experiment'] = pd.Categorical(_df_agg['experiment'], categories=relevant_experiments, ordered=True)
    _df_agg = _df_agg.sort_values(by=['experiment', 'query_ident'])

    # Assign a numeric index based on the unique 'query_experiment' to serve as the x-axis
    unique_queries = _df_agg['query_experiment'].unique()
    query_index = {query: idx for idx, query in enumerate(unique_queries)}
    _df_agg['query_index'] = _df_agg['query_experiment'].apply(lambda x: query_index[x])

    # Identify all methods excluding PostgreSQL
    methods = _df_agg['method'].unique()
    methods = [m for m in methods if m != 'PostgreSQL']

    # Determine number of rows for subplots
    num_rows = len(methods)
    fig, axes = plt.subplots(num_rows, 1, figsize=(15, 2 * num_rows))

    # Ensure 'axes' is iterable
    if num_rows == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        # Pivot the DataFrame to get each method and PostgreSQL as columns with their respective total times
        pivot_df = _df_agg.pivot_table(index='query_index', columns='method', values=time_measurement, aggfunc='mean')
        # Plot PostgreSQL and the current method
        ax.plot(pivot_df.index, pivot_df['PostgreSQL'], label="PostgreSQL", color='blue')
        if method in pivot_df.columns:
            ax.plot(pivot_df.index, pivot_df[method], label=method, color='red')

        # Add vertical lines at transitions between different workloads
        previous_experiment = None
        for idx, query in enumerate(unique_queries):
            current_experiment = query.split('____')[0]
            if previous_experiment and current_experiment != previous_experiment:
                ax.axvline(x=idx - 0.5, color='black', linestyle='--')
            previous_experiment = current_experiment

        ax.set_ylabel('Mean Total Time (ms)')
        ax.set_yscale("log")  # Set y-axis to logarithmic scale
        ax.legend(ncol=5)
        ax.grid(True)

    plt.tight_layout()  # Adjust layout to make room for subplots
    plt.savefig(f'combined_plot_{time_measurement}.pdf', format='pdf')  # Save the entire figure as a PDF
    plt.close()
    exit(0)


# In[4]:


def query_file_to_ident(file_name):
    ident = file_name.split('.sql')[0]
    return f"{ident[:-1].zfill(2)}{ident[-1]}"


TEST_QUERIES = dict()

for folder in os.listdir('bao/queries'):
    if not folder.startswith('job__'):
        continue

    key = folder.split('job__')[1]

    queries = os.listdir(os.path.join('bao', 'queries', folder, 'test'))
    queries = sorted([query_file_to_ident(x) for x in queries])

    TEST_QUERIES[key] = queries

for k in sorted(TEST_QUERIES.keys()):
    print(k)
    print(TEST_QUERIES[k])
    print('---' * 30)

# ### Training Time of Different Methods

# In[5]:


training_time = {
    'Bao': {
        'base_query_split_1': 1.3,
        'base_query_split_2': 1.2,
        'base_query_split_3': 1.3,
        'leave_one_out_split_1': 1.4,
        'leave_one_out_split_2': 1.2,
        'leave_one_out_split_3': 1.3,
        'random_split_1': 1.2,
        'random_split_2': 1.3,
        'random_split_3': 1.1,
    },
    'Neo': {
        'base_query_split_1': 37.1,
        'base_query_split_2': 46.4,
        'base_query_split_3': 23.1,
        'leave_one_out_split_1': 40.6,
        'leave_one_out_split_2': 39.3,
        'leave_one_out_split_3': 19.1,
        'random_split_1': 20.5,
        'random_split_2': 17.1,
        'random_split_3': 17.9,
    },
    'Balsa': {
        'base_query_split_1': 86.4,
        'base_query_split_2': 40.2,
        'base_query_split_3': 63.8,
        'leave_one_out_split_1': 56.9,
        'leave_one_out_split_2': 66.1,
        'leave_one_out_split_3': 56.1,
        'random_split_1': 38.8,
        'random_split_2': 55.0,
        'random_split_3': 44.7,
    },
    'LEON': {
        'base_query_split_1': 118.1,
        'base_query_split_2': 125.5,
        'base_query_split_3': 124.1,
        'leave_one_out_split_1': 116.5,
        'leave_one_out_split_2': 114.1,
        'leave_one_out_split_3': 112.5,
        'random_split_1': 113.0,
        'random_split_2': 115.7,
        'random_split_3': 120.1,
    },
    'HybridQO': {
        'base_query_split_1': 18.7,
        'base_query_split_2': 21.5,
        'base_query_split_3': 17,
        'leave_one_out_split_1': 21.5,
        'leave_one_out_split_2': 19.8,
        'leave_one_out_split_3': 17.2,
        'random_split_1': 18.3,
        'random_split_2': 19.67,
        'random_split_3': 17.7,
    },
    'PostgreSQL': {
        'base_query_split_1': 0,
        'base_query_split_2': 0,
        'base_query_split_3': 0,
        'leave_one_out_split_1': 0,
        'leave_one_out_split_2': 0,
        'leave_one_out_split_3': 0,
        'random_split_1': 0,
        'random_split_2': 0,
        'random_split_3': 0,
    }
}

df_training = pd.DataFrame(training_time)
df_training

# In[6]:


df_train_agg = df_training.sum().reset_index()
# renaming
df_train_agg.columns = ['method', 'execution_time__sum_h']
df_train_agg['execution_time__sum_d'] = df_train_agg['execution_time__sum_h'] / 24.0
df_train_agg

# ### Reading all result files

# In[7]:


EXECUTION_TIME_OUT = 360000.0

df = pd.read_csv('experiment_logs/20240103__combined.csv')
df['prepare_time'] = df['inference_time'] + df['planning_time']
display(df)

#  check the datasets
df_test = df[df["method"] == "PostgreSQL"]
df_test = df_test[df_test["run_type"] == "regular"]

# ### Create aggregated dataframe
# Iterate over all experiments ('split types') and aggregate the numbers across the various runs. Then make the same for PostgreSQL and copy each PostgreSQL run for all these split types, such that the train/test split is applied to make further comparisons easier (even though of course PostgreSQL is not trained in any capacity).

# In[8]:

"""
run_id, run_type, experiment (worklaod), method (system), query_ident (q_id), split,inference_time, planning_time, execution_time, total_time, timed_out
0, regular, base_query_split_1, Balsa, 01a, train, 52.400000000000006, 0.719, 986.883, 1040.002, False
"""

aggregated_dfs = []
split_df = \
    df.groupby(['experiment', 'query_ident', 'split']).count().reset_index().sort_values(['experiment', 'query_ident'])[
        ['experiment', 'query_ident', 'split']]

for (method, run_type, experiment), df_group in df.groupby(['method', 'run_type', 'experiment'], dropna=False):
    if run_type != 'regular':
        continue

    print("debug: ", (method, run_type, experiment))

    df_group.loc[df_group.index, 'timed_out'] = df_group.loc[df_group.index, 'timed_out'].astype(int)

    x = df_group.groupby(['method', 'run_type', 'experiment', 'query_ident', 'split'], dropna=False)
    aggregated = x.agg(
        {'inference_time': ['mean', 'std'], 'planning_time': ['mean', 'std'], 'execution_time': ['mean', 'std'],
         'total_time': ['mean', 'std'], 'prepare_time': ['mean', 'std'], 'timed_out': 'max'}).reset_index()

    # Fix nested names from double aggregations (mean+std)
    new_col_names = []
    for c in aggregated.columns:
        if c[1] == 'std':
            new_col_names.append(".".join(c))  # for standard deviation
        else:
            new_col_names.append(c[0])  # for mean values

    aggregated.columns = new_col_names

    aggregated = aggregated.sort_values(['query_ident'])

    aggregated.loc[aggregated['timed_out'] > 0, 'timed_out'] = 1.0
    aggregated.loc[aggregated['timed_out'] > 0, 'execution_time'] = EXECUTION_TIME_OUT
    # Add correction to total_time
    aggregated['total_time'] = aggregated['inference_time'] + aggregated['planning_time'] + aggregated['execution_time']

    # Add number of runs per query
    aggregated['n'] = x.count()['run_id'].tolist()

    if method != 'PostgreSQL':
        if 'leave_one_out' in experiment:
            aggregated['base_experiment'] = 'leave_one_out'
        elif 'base_query_split' in experiment:
            aggregated['base_experiment'] = 'base_query'
        elif 'random_split' in experiment:
            aggregated['base_experiment'] = 'random'

        aggregated_dfs.append(aggregated.copy())

    else:
        # new_experiment is a string
        for new_experiment in df['experiment'].dropna().unique():
            tmp = aggregated.copy()
            tmp = tmp.sort_values(['query_ident'])

            tmp['experiment'] = new_experiment
            tmp['split'] = split_df[split_df['experiment'] == new_experiment]['split'].tolist()

            if 'leave_one_out' in new_experiment:
                tmp['base_experiment'] = 'leave_one_out'
            elif 'base_query_split' in new_experiment:
                tmp['base_experiment'] = 'base_query'
            elif 'random_split' in new_experiment:
                tmp['base_experiment'] = 'random'

            aggregated_dfs.append(tmp)

df_agg = pd.concat(aggregated_dfs)
# display(df_agg.groupby(['method', 'experiment']).count())
display(df_agg)

df_agg_test = df_agg[df_agg["method"] == "PostgreSQL"]
df_unique_df = df_agg_test.drop_duplicates(subset='query_ident', keep='first')

draw_query_by_query_bar_comapre_with_me(df_agg)
draw_query_by_query_multiple_lines_comapre_with_me(df_agg)
draw_query_by_query_bar_charts(df_agg)
draw_query_by_query_multiple_lines(df_agg)
draw_query_by_query(df_agg)
exit(0)

# ### Calculate result table values

# In[9]:

# sum over all queries.
df_sum = df_agg \
    .groupby(['base_experiment', 'experiment', 'method', 'split']) \
    .agg({
    'total_time': 'sum', 'execution_time': 'sum', 'planning_time': 'sum', 'inference_time': 'sum',
    'prepare_time': 'sum',
    'total_time.std': 'sum', 'execution_time.std': 'sum', 'planning_time.std': 'sum', 'inference_time.std': 'sum',
    'prepare_time.std': 'sum',
    'n': 'mean'
}) \
    .reset_index()

#  1.96 is the critical value for a 95% confidence interval (assuming a normal distribution),
df_sum['execution_time.err'] = 1.96 * df_sum['execution_time.std'] / np.sqrt(df_sum['n'])
df_sum['planning_time.err'] = 1.96 * df_sum['planning_time.std'] / np.sqrt(df_sum['n'])
df_sum['inference_time.err'] = 1.96 * df_sum['inference_time.std'] / np.sqrt(df_sum['n'])
df_sum['total_time.err'] = 1.96 * df_sum['total_time.std'] / np.sqrt(df_sum['n'])
df_sum['prepare_time.err'] = 1.96 * df_sum['prepare_time.std'] / np.sqrt(df_sum['n'])

df_sum.iloc[:5]

df_sum_test = df_sum[df_sum["method"] == "PostgreSQL"]

# # Results Figure: Test Set

# In[11]:


methods = ['PostgreSQL', 'Neo', 'Bao', 'Balsa', 'LEON', 'HybridQO']
experiments = [
                  'leave_one_out_split_1', 'leave_one_out_split_2', 'leave_one_out_split_3',
                  'random_split_1', 'random_split_2', 'random_split_3',
                  'base_query_split_1', 'base_query_split_2', 'base_query_split_3',
              ][::-1]

experiment_labels = [
                        'Leave One Out Split 1\n(LOO-1)', 'Leave One Out Split 2\n(LOO-2)',
                        'Leave One Out Split 3\n(LOO-3)',
                        'Random Split 1\n(RND-1)', 'Random Split 2\n(RND-2)', 'Random Split 3\n(RND-3)',
                        'Base Query Split 1\n(BQ-1)', 'Base Query Split 2\n(BQ-2)', 'Base Query Split 3\n(BQ-3)',
                    ][::-1]

experiment_labels__short = ['LOO-1', 'LOO-2', 'LOO-3', 'RND-1', 'RND-2', 'RND-3', 'BQ-1', 'BQ-2', 'BQ-3'][::-1]

plot_colors = np.array(list(mcolors.TABLEAU_COLORS.keys()))

split = 'test'
bar_height = (1 / (len(methods) + 1))

plt.figure(figsize=(14, 9))

#
# ===============================================================================================================================
#  Planning & Inference Time Figure
# ===============================================================================================================================
#
ax_left = plt.subplot2grid((1, 35), (0, 0), colspan=9, rowspan=1)
ax_right = plt.subplot2grid((1, 35), (0, 10), colspan=6, rowspan=1)

planning_handles = []
inference_handles = []
for mi, method in enumerate(methods):
    df_exp = df_sum[(df_sum['split'] == split) & (df_sum['method'] == method)]

    bar_pos = np.arange(len(experiments)) + ((len(methods) / 2) * bar_height) - (mi * bar_height) - bar_height
    bar_color = plot_colors[mi]

    # Add Planning Bars
    # ===============================================
    planning_data = np.array(
        [df_exp.loc[df_exp['experiment'] == e, 'planning_time'].item() for e in experiments]) / 1000.0

    # Special handling of Bao, as their inference time is part of the planning time reported by PostgreSQL
    if method == 'Bao':
        barh = ax_left.barh(bar_pos, planning_data, height=bar_height, label=method, align='edge', color=bar_color,
                            zorder=1, facecolor='none')

        faded_color = plt.get_cmap('tab10')(mi)[:-1] + (0.5,)
        cmap = LinearSegmentedColormap.from_list('CustomColormap', [plt.get_cmap('tab10')(mi), faded_color], N=256)
        grad = np.atleast_2d(
            np.linspace(0, 1, 16))  # last number defines amount of gradient steps, low to keep it visible
        for bar in barh:
            x, y = bar.get_xy()
            w = bar.get_width()
            h = bar.get_height()

            ax_left.imshow(grad, extent=[x, x + w, y, y + h], cmap=cmap, aspect='auto', zorder=0)

        # To add Bao's legend back
        barh = ax_left.barh(bar_pos, [0 for _ in range(len(experiments))], label=method, color=bar_color)
        ax_right.barh(bar_pos, [0 for _ in range(len(experiments))], label=method, color=bar_color)

    else:
        barh = ax_left.barh(bar_pos, planning_data, height=(1 / (len(methods) + 1)), label=method, align='edge',
                            color=bar_color)
        barh = ax_right.barh(bar_pos, planning_data, height=(1 / (len(methods) + 1)), label=method, align='edge',
                             color=bar_color)

    planning_handles.append(barh)

    # Add Inference Bars
    # ===============================================
    inference_data = np.array(
        [df_exp.loc[df_exp['experiment'] == e, 'inference_time'].item() for e in experiments]) / 1000.0

    planning_std = np.array(
        [df_exp.loc[df_exp['experiment'] == e, 'planning_time.err'].item() for e in experiments]) / 1000.0
    inference_std = np.array(
        [df_exp.loc[df_exp['experiment'] == e, 'inference_time.err'].item() for e in experiments]) / 1000.0
    xerr = np.array([np.sqrt(p ** 2 + i ** 2) for (p, i) in zip(planning_std, inference_std)])

    barh = ax_left.barh(bar_pos, inference_data, xerr=xerr, capsize=5, left=planning_data,
                        height=(1 / (len(methods) + 1)), align='edge', color=bar_color, alpha=0.5)
    ax_right.barh(bar_pos, inference_data, xerr=xerr, capsize=5, left=planning_data, height=(1 / (len(methods) + 1)),
                  align='edge', color=bar_color, alpha=0.5)
    inference_handles.append(barh)

ax_left.set_yticks(range(len(experiment_labels)))
ax_left.set_yticklabels(experiment_labels)
ax_left.tick_params(axis='x', pad=6)
ax_right.yaxis.tick_right()
ax_right.set_yticks([])

ax_left.spines['right'].set_visible(False)
ax_right.spines['left'].set_visible(False)

ax_left.set_xlim([0, 18])
ax_right.set_xlim([20, 10 ** 5 * 1.2])
ax_right.set_xscale('log')
ax_right.set_xticks([10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5])

ax_right.xaxis.set_minor_locator(LogLocator(base=10, numticks=12, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)))

# Add the diagonal markings to show that the x-axis is broken
d = 0.015
kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False)
ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
kwargs.update(transform=ax_right.transAxes)  # switch to the bottom axes
angle_scale = 1.5
ax_right.plot((-d * angle_scale, +d * angle_scale), (1 - d, 1 + d), **kwargs)
ax_right.plot((-d * angle_scale, +d * angle_scale), (-d, +d), **kwargs)

# Set up legend
legend_handles = [(p, h) for p, h in zip(planning_handles, inference_handles)]
legend_labels = [f"{m}: Planning / Inference Time" for m in methods]
ax_left.legend(legend_handles, legend_labels, handler_map={tuple: HandlerTuple(ndivide=None)},
               bbox_to_anchor=(0.58, -0.07), ncol=2, loc='upper center')

ax_left.set_ylim([-0.5, len(experiments) - 0.5])
ax_right.set_ylim([-0.5, len(experiments) - 0.5])
ax_left.text(9, -1.06, 'Planning and Inference Time [s]', fontsize=10)

#
# ===============================================================================================================================
#  End-to-End Execution Time Figure
# ===============================================================================================================================
#
ax_left = plt.subplot2grid((1, 35), (0, 19), colspan=10, rowspan=1)
ax_right = plt.subplot2grid((1, 35), (0, 30), colspan=5, rowspan=1)

execution_handles = []
for mi, method in enumerate(methods):
    df_exp = df_sum[(df_sum['split'] == split) & (df_sum['method'] == method)]

    bar_pos = np.arange(len(experiments)) + ((len(methods) / 2) * bar_height) - (mi * bar_height) - bar_height
    bar_color = plot_colors[mi]

    # Add Execution Bars
    # ===============================================
    execution_data = np.array(
        [df_exp.loc[df_exp['experiment'] == e, 'total_time'].item() for e in experiments]) / 1000.0

    execution_std = np.array(
        [df_exp.loc[df_exp['experiment'] == e, 'total_time.err'].item() for e in experiments]) / 1000.0
    xerr = execution_std

    barh = ax_left.barh(bar_pos, execution_data, xerr=xerr, capsize=5, height=(1 / (len(methods) + 1)), align='edge',
                        color=bar_color)
    barh = ax_right.barh(bar_pos, execution_data, xerr=xerr, capsize=5, height=(1 / (len(methods) + 1)), align='edge',
                         color=bar_color)
    execution_handles.append(barh)

# Set up legend
ax_left.legend(execution_handles, methods, bbox_to_anchor=(0.75, -0.07), ncol=2, loc='upper center')

ax_left.spines['right'].set_visible(False)
ax_right.spines['left'].set_visible(False)

# Add the diagonal markings to show that the x-axis is broken
d = 0.015
kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False)
ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
kwargs.update(transform=ax_right.transAxes)  # switch to the bottom axes
angle_scale = 1.9
ax_right.plot((-d * angle_scale, +d * angle_scale), (1 - d, 1 + d), **kwargs)
ax_right.plot((-d * angle_scale, +d * angle_scale), (-d, +d), **kwargs)

ax_left.set_yticklabels([None] + experiment_labels__short)
ax_left.tick_params(axis='x', pad=5.8)
ax_right.yaxis.tick_right()
ax_right.set_yticks([])

ax_left.set_xlim([0, 83])
ax_right.set_xlim([85, 10 ** 5 * 1.2])
ax_right.set_xscale('log')
ax_right.set_xticks([10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5])

ax_right.xaxis.set_minor_locator(LogLocator(base=10, numticks=12, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)))

ax_left.set_ylim([-0.5, len(experiments) - 0.5])
ax_right.set_ylim([-0.5, len(experiments) - 0.5])

ax_left.text(40, -1.06, 'End-to-End Execution Time [s]', fontsize=10)

plt.subplots_adjust(wspace=.6)
plt.show()
None
