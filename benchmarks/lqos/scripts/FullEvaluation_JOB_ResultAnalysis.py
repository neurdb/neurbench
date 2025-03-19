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

# # Results Figure: Training Set

# In[10]:


# methods = ['LEON', 'Balsa', 'Bao', 'Neo', 'PostgreSQL'] #reversed
methods = ['PostgreSQL', 'Neo', 'Bao', 'Balsa', 'LEON', 'HybridQO']
experiments = [
                  'leave_one_out_split_1', 'leave_one_out_split_2', 'leave_one_out_split_3',
                  'random_split_1', 'random_split_2', 'random_split_3',
                  'base_query_split_1', 'base_query_split_2', 'base_query_split_3',
              ][::-1]

experiment_labels = [
                        'Leave One Out Split 1', 'Leave One Out Split 2', 'Leave One Out Split 3',
                        'Random Split 1', 'Random Split 2', 'Random Split 3',
                        'Base Query Split 1', 'Base Query Split 2', 'Base Query Split 3',
                    ][::-1]

experiment_labels__short = ['LOO-1', 'LOO-2', 'LOO-3', 'RND-1', 'RND-2', 'RND-3', 'BQ-1', 'BQ-2', 'BQ-3'][::-1]

plot_colors = np.array(list(mcolors.TABLEAU_COLORS.keys()))

split = 'train'
bar_height = (1 / (len(methods) + 1))

# fig, axs = plt.subplots(1, 2, figsize=(10, 9))
plt.figure(figsize=(14, 9))

# PLANNING + INFERENCE PLOT
# =================================================================================
ax = plt.subplot2grid((1, 13), (0, 0), colspan=6, rowspan=1)

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
        barh = ax.barh(bar_pos, planning_data, height=bar_height, label=method, align='edge', color=bar_color, zorder=1,
                       facecolor='none')

        faded_color = plt.get_cmap('tab10')(mi)[:-1] + (0.5,)
        cmap = LinearSegmentedColormap.from_list('CustomColormap', [plt.get_cmap('tab10')(mi), faded_color], N=256)
        grad = np.atleast_2d(
            np.linspace(0, 1, 16))  # last number defines amount of gradient steps, low to keep it visible
        for bar in barh:
            x, y = bar.get_xy()
            w = bar.get_width()
            h = bar.get_height()

            ax.imshow(grad, extent=[x, x + w, y, y + h], cmap=cmap, aspect='auto', zorder=0)

        # To add Bao's legend back
        barh = ax.barh(bar_pos, [0 for _ in range(len(experiments))], label=method, color=bar_color)

    else:
        barh = ax.barh(bar_pos, planning_data, height=(1 / (len(methods) + 1)), label=method, align='edge',
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

    barh = ax.barh(bar_pos, inference_data, xerr=xerr, capsize=5, left=planning_data, height=(1 / (len(methods) + 1)),
                   align='edge', color=bar_color, alpha=0.5)
    inference_handles.append(barh)

# Set up legend
legend_handles = [(p, h) for p, h in zip(planning_handles, inference_handles)]
legend_labels = [f"{m}: Planning / Inference Time" for m in methods]
ax.legend(legend_handles, legend_labels, handler_map={tuple: HandlerTuple(ndivide=None)}, bbox_to_anchor=(0.37, -0.07),
          ncol=2, loc='upper center')

ax.set_yticks(np.arange(len(experiments)))
ax.set_yticklabels(experiment_labels)

ax.set_ylim([-0.5, len(experiments) - 0.5])
ax.set_xlabel('Planning and Inference Time [s]')
ax.set_xscale('log')

# EXECUTION PLOT
# =================================================================================
ax_left = plt.subplot2grid((1, 13), (0, 7), colspan=3, rowspan=1)
ax_right = plt.subplot2grid((1, 13), (0, 10), colspan=3, rowspan=1)

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
ax_left.legend(execution_handles, methods, bbox_to_anchor=(1.1, -0.07), ncol=2, loc='upper center')

ax_left.spines['right'].set_visible(False)
ax_right.spines['left'].set_visible(False)

# Add the diagonal markings to show that the x-axis is broken
d = 0.015
kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False)
ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
kwargs.update(transform=ax_right.transAxes)  # switch to the bottom axes
ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_right.plot((-d, +d), (-d, +d), **kwargs)

ax_left.set_yticklabels(['LOO-1', 'LOO-2', 'LOO-3', 'RND-1', 'RND-2', 'RND-3', 'BQ-1', 'BQ-2', 'BQ-3', None][::-1])
ax_right.yaxis.tick_right()
ax_right.set_yticks([])
ax_right.set_xticks([500, 2500, 4500, 6500, 8500])

ax_left.set_xlim([0, 300])
ax_right.set_xlim([500, 8500])

ax_left.set_ylim([-0.5, len(experiments) - 0.5])
ax_right.set_ylim([-0.5, len(experiments) - 0.5])

ax_left.text(180, -1.02, 'End-to-End Execution Time [s]', fontsize=10)

plt.subplots_adjust(wspace=.6)

plt.show()
None

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


# ## Inference + Planning vs. Execution Time on all splits

# In[12]:


def plot_scatter_execution(ax, df, min_val, max_val, legend_values, title):
    # Plot non timed out queries (majority)
    df_train = df[(df['split'] == 'train') & (df['timed_out'] == 0)]
    df_test = df[(df['split'] == 'test') & (df['timed_out'] == 0)]
    tr1 = ax.scatter(df_train['execution_time'], df_train['planning_time'] + df_train['inference_time'],
                     color='tab:blue')
    te1 = ax.scatter(df_test['execution_time'], df_test['planning_time'] + df_test['inference_time'],
                     color='tab:orange')

    # Show timed out queries a "X" markers instead of circles
    df_train = df[(df['split'] == 'train') & (df['timed_out'] > 0)]
    df_test = df[(df['split'] == 'test') & (df['timed_out'] > 0)]
    tr2 = ax.scatter(df_train['execution_time'], df_train['planning_time'] + df_train['inference_time'],
                     color='tab:blue', marker='x')
    te2 = ax.scatter(df_test['execution_time'], df_test['planning_time'] + df_test['inference_time'],
                     color='tab:orange', marker='x')

    ax.set_xlabel('Execution Time [ms]')
    ax.set_ylabel('Inference + Planning Time [ms]')

    l, = ax.plot([min_val, max_val], [min_val, max_val], color='tab:gray', linestyle='-.')

    ax.set_xlim([min_val, max_val])
    ax.set_xscale('log')

    ax.set_ylim([min_val, max_val])
    ax.set_yscale('log')

    legend = ax.legend([(tr1, tr2), (te1, te2), l], legend_values, handler_map={tuple: HandlerTuple(ndivide=None)})

    ax.set_title(title)
    ax.grid(True)


# In[14]:


experiments = ['leave_one_out_split_1', 'leave_one_out_split_2', 'leave_one_out_split_3',
               'random_split_1', 'random_split_2', 'random_split_3',
               'base_query_split_1', 'base_query_split_2', 'base_query_split_3']


def experiment_label(experiment_name):
    if 'leave_one_out' in experiment_name:
        return f'Leave One Out (Split #{experiment_name[-1]})'
    elif 'random' in experiment_name:
        return f'Random (Split #{experiment_name[-1]})'
    elif 'base_query' in experiment_name:
        return f'Base Query (Split #{experiment_name[-1]})'
    else:
        return ''


num_experiments = len(experiments)
num_methods = 5
plot_scale = 5
legend_values = ['Training Set Queries', 'Test Set Queries', 'Inference + Planning Time == Execution Time']

fig, axs = plt.subplots(num_experiments, num_methods, figsize=(plot_scale * num_methods, plot_scale * num_experiments))
min_val = df_agg['total_time'].min() * 0.8
max_val = df_agg['total_time'].max() * 1.2

for m_id, method in enumerate(['PostgreSQL', 'Neo', 'Bao', 'Balsa', 'LEON']):
    for e_id, experiment in enumerate(experiments):
        ax = axs[e_id, m_id]
        plot_scatter_execution(ax, df_agg[(df_agg['method'] == method) & (df_agg['experiment'] == experiment)],
                               min_val=min_val, max_val=max_val, legend_values=legend_values,
                               title=f"{method}: {experiment_label(experiment)}")

plt.tight_layout()
plt.show()
None


# ### Ablation Study: Disabling BitmapScan / TidScan

# In[15]:


def comparison_plot(df, df_nobit, threshold, legend_handles, show_overview=True, ymax_override=None):
    ymax = max(
        df['execution_time'].max(),
        df_nobit['execution_time'].max()
    ) * 1.05

    df = df.sort_values(['query_ident'])
    df_nobit = df_nobit.sort_values(['query_ident'])

    query_labels = np.array(df['query_ident'].unique())

    pg_execution_time = np.array(df.groupby(['query_ident']).mean(numeric_only=True)['execution_time'])
    nobit_execution_time = np.array(df_nobit.groupby(['query_ident']).mean(numeric_only=True)['execution_time'])

    pg_err = np.array(df.groupby(['query_ident']).std(numeric_only=True)['execution_time']) / np.sqrt(
        df.groupby(['query_ident']).count()['run_id'].tolist())
    nobit_err = np.array(df_nobit.groupby(['query_ident']).std(numeric_only=True)['execution_time']) / np.sqrt(
        df_nobit.groupby(['query_ident']).count()['run_id'].tolist())
    combined_err = 1.96 * np.array([np.sqrt(s1 ** 2 + s2 ** 2) for s1, s2 in zip(pg_err, nobit_err)])

    diff = np.array(pg_execution_time - nobit_execution_time)
    pg_diff = np.array([d if d < 0 else 0 for d in diff])
    nobit_diff = np.array([d if d > 0 else 0 for d in diff])

    if show_overview:
        # General overview plot
        fig, axs = plt.subplots(3, figsize=(20, 12))

        axs[0].bar(query_labels, pg_execution_time, yerr=pg_err, capsize=2.5)
        axs[0].set_title(
            f'PostgreSQL: All Scans enabled\n(Total Workload Execution Time: {pg_execution_time.sum() / 1000:.2f} seconds)')
        axs[0].set_ylim(top=ymax)
        axs[0].set_ylabel('Execution Time [ms]')

        axs[1].bar(query_labels, nobit_execution_time, yerr=nobit_err, capsize=2.5)
        axs[1].set_title(
            f'PostgreSQL: No BitmapScan, No TidScan\n(Total Workload Execution Time: {nobit_execution_time.sum() / 1000:.2f} seconds)')
        axs[1].set_ylim(top=ymax)
        axs[1].set_ylabel('Execution Time [ms]')

        axs[2].bar(query_labels, pg_diff, color='tab:blue', label='a')
        axs[2].bar(query_labels, nobit_diff, color='tab:orange', label='b')
        axs[2].bar(query_labels, [0 for _ in range(len(query_labels))], yerr=combined_err, label=None, capsize=2.5)
        axs[2].set_title('Execution Time Difference')
        axs[2].legend(legend_handles)
        axs[2].set_ylabel('Difference in Execution Time [ms]')

        for ax in axs:
            ax.set_xlim([-0.5, len(query_labels) - 0.5])
            ax.set_xticklabels(query_labels, rotation=90)

        plt.tight_layout()
        plt.show()

    # Filtered plot
    mask = np.array((diff > threshold) | (diff < -threshold))

    pg_diff = np.array([d if d < 0 else 0 for d in diff[mask]])
    pg_diff_significant = np.array([d if d <= -e else 0 for d, e in zip(pg_diff, combined_err[mask])])
    pg_diff_not_significant = np.array([d if d > -e else 0 for d, e in zip(pg_diff, combined_err[mask])])

    nobit_diff = np.array([d if d > 0 else 0 for d in diff[mask]])
    nobit_diff_significant = np.array([d if d >= e else 0 for d, e in zip(nobit_diff, combined_err[mask])])
    nobit_diff_not_significant = np.array([d if d < e else 0 for d, e in zip(nobit_diff, combined_err[mask])])

    fig, ax = plt.subplots(1, figsize=(7, 3.2))

    p1 = ax.bar(query_labels[mask], - pg_diff_significant / 1000.0, color='tab:blue', label='a')
    n1 = ax.bar(query_labels[mask], nobit_diff_significant / 1000.0, color='tab:orange', label='b')

    p2 = ax.bar(query_labels[mask], - pg_diff_not_significant / 1000.0, color='tab:blue', alpha=0.5)
    n2 = ax.bar(query_labels[mask], nobit_diff_not_significant / 1000.0, color='tab:orange', alpha=0.5)

    e = ax.bar(query_labels[mask], [0 for _ in range(np.sum(mask))], yerr=combined_err[mask] / 1000.0, label=None,
               capsize=4)
    # ax.set_title('Execution Time Difference')
    # ax.legend(['Faster with BitmapScan/TidScan enabled', 'Faster with BitmapScan/TidScan disabled'])

    legend = ax.legend([(p1, p2), (n1, n2), e], legend_handles, handler_map={tuple: HandlerTuple(ndivide=None)})

    ax.set_ylabel('Mean Difference in Execution Time [s]')

    ax.set_ylim(bottom=0)
    ax.set_xlim([-0.5, np.sum(mask) - 0.5])
    ax.set_xticklabels(query_labels[mask], rotation=90)

    if ymax_override is not None:
        ax.set_ylim(top=ymax_override)

    plt.tight_layout()
    plt.show()


# In[16]:


df_pg = df[(df['method'] == 'PostgreSQL') & (df['run_type'] == 'regular')]
df_nobit = df[(df['method'] == 'PostgreSQL') & (df['run_type'] == 'no_bitmap_tidscan')]

comparison_plot(df_pg, df_nobit, threshold=250, legend_handles=['Faster with Bitmap and Tid Scans Enabled',
                                                                'Faster with Bitmap and Tid Scans Disabled'],
                show_overview=False)


# In[17]:


def create_df(df, df_nobit, threshold):
    query_labels = np.array(df['query_ident'].unique())

    pg_execution_time = np.array(df.groupby(['query_ident']).mean(numeric_only=True)['execution_time'])
    nobit_execution_time = np.array(df_nobit.groupby(['query_ident']).mean(numeric_only=True)['execution_time'])

    pg_planning_time = np.array(df.groupby(['query_ident']).mean(numeric_only=True)['planning_time'])
    nobit_planning_time = np.array(df_nobit.groupby(['query_ident']).mean(numeric_only=True)['planning_time'])

    pg_err = np.array(df.groupby(['query_ident']).std(numeric_only=True)['execution_time'])
    nobit_err = np.array(df_nobit.groupby(['query_ident']).std(numeric_only=True)['execution_time'])
    combined_err = np.array([np.sqrt(s1 ** 2 + s2 ** 2) for s1, s2 in zip(pg_err, nobit_err)])

    diff = np.array(pg_execution_time - nobit_execution_time)
    pg_diff = np.array([d if d < 0 else 0 for d in diff])
    nobit_diff = np.array([d if d > 0 else 0 for d in diff])

    df_combined = pd.DataFrame({
        'query_ident': query_labels,
        'execution_time_pg': pg_execution_time,
        'execution_time_nobit': nobit_execution_time,
        'planning_time_pg': pg_planning_time,
        'planning_time_nobit': nobit_planning_time,
        'diff': diff,
        'err': combined_err,
    })

    df_combined = df_combined[(df_combined['diff'] > threshold) | (df_combined['diff'] < -threshold)]
    df_combined = df_combined.sort_values(['query_ident']).reset_index(drop=True)
    df_combined = df_combined[
        ['query_ident', 'planning_time_pg', 'execution_time_pg', 'planning_time_nobit', 'execution_time_nobit', 'diff',
         'err']]

    df_combined['%diff_nobit'] = df_combined['execution_time_pg'] / df_combined['execution_time_nobit']
    df_combined['%diff_pg'] = df_combined['execution_time_nobit'] / df_combined['execution_time_pg']
    df_combined.sort_values(['%diff_pg'])

    df_combined['significant'] = df_combined['diff'].abs() > df_combined['err'].abs()
    return df_combined


df_combined = create_df(df_pg, df_nobit, threshold=250)
df_combined[df_combined['significant']].sort_values(['%diff_nobit'])

# ### Ablation Study: Disabling GEQO
# (The `_no_bit` suffix here means no GEQO, to reuse the code from above with the disabling of bitmap/tid scans)

# In[18]:


df_pg = df[(df['method'] == 'PostgreSQL') & (df['run_type'] == 'regular')]
df_nogeqo = df[(df['method'] == 'PostgreSQL') & (df['run_type'] == 'no_geqo')]

comparison_plot(df_pg, df_nogeqo, threshold=100,
                legend_handles=['Faster with GEQO Enabled', 'Faster with GEQO Disabled'], show_overview=False)

# In[19]:


df_combined = create_df(df_pg, df_nogeqo, threshold=100)
df_combined[df_combined['significant']].sort_values(['%diff_nobit'])

# ### Ablation Study: IMDB50 Trained Bao

# In[20]:


import warnings

import pandas as pd
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def process_imdb50_df(df):
    df = df.copy()
    df = df.groupby(['query_ident']).agg({
        # 'inference_time': ['mean', 'std'],
        # 'planning_time': ['mean', 'std'],
        # 'execution_time': ['mean', 'std'],
        'total_time': 'mean',
        'run_id': 'count'
    })
    df = df.sort_values(['query_ident']).reset_index()

    if (df['total_time'] < 0.001).any() and (df['total_time'] > 0).any():
        print(df[df['total_time'] < 0.001])

    return df


pg_raw = df[(df['method'] == 'PostgreSQL') & (df['run_type'] == 'regular') & (df['run_id'].isin([0, 1, 2, 3]))]
bao_raw = df[(df['method'] == 'Bao') & (df['experiment'] == 'imdb50_trained__imdb_tested')]
pg50_raw = df[(df['method'] == 'PostgreSQL') & (df['run_type'] == 'imdb50')]
bao50_raw = df[(df['method'] == 'Bao') & (df['experiment'] == 'imdb50_trained__imdb50_tested')]

pg_raw['one_over_total_time_squared'] = (1 / pg_raw['total_time']) ** 2
pg50_raw['one_over_total_time_squared'] = (1 / pg50_raw['total_time']) ** 2

bao_raw['total_time_squared'] = bao_raw['total_time'] ** 2
bao50_raw['total_time_squared'] = bao50_raw['total_time'] ** 2

pg_raw['1_over_total_time'] = 1 / pg_raw['total_time']
pg50_raw['1_over_total_time'] = 1 / pg50_raw['total_time']

pg = pg_raw.groupby(['query_ident']).mean(numeric_only=True)['total_time']
bao = bao_raw.groupby(['query_ident']).mean(numeric_only=True)['total_time']

pg50 = pg50_raw.groupby(['query_ident']).mean(numeric_only=True)['total_time']
bao50 = bao50_raw.groupby(['query_ident']).mean(numeric_only=True)['total_time']

one_over_pg_2 = pg_raw.groupby(['query_ident']).mean(numeric_only=True)['one_over_total_time_squared']
one_over_pg50_2 = pg50_raw.groupby(['query_ident']).mean(numeric_only=True)['one_over_total_time_squared']

bao_2 = bao_raw.groupby(['query_ident']).mean(numeric_only=True)['total_time_squared']
bao50_2 = bao50_raw.groupby(['query_ident']).mean(numeric_only=True)['total_time_squared']

one_over_pg = pg_raw.groupby(['query_ident']).mean(numeric_only=True)['1_over_total_time']
one_over_pg50 = pg50_raw.groupby(['query_ident']).mean(numeric_only=True)['1_over_total_time']

# Full IMDB Expectation + Variance for Bao/Postgres Performance
imdb = bao * (one_over_pg)
imdb_var = ((bao_2) * (one_over_pg_2) - (imdb ** 2))

# IMDB 50% Expectation + Variance for Bao/Postgres Performance
imdb50 = bao50 * (one_over_pg50)
imdb50_var = (bao50_2 * (one_over_pg50_2) - (imdb50 ** 2))

# In[21]:


top_n = 30
y_max = 25

idx = np.argsort(np.abs(imdb - imdb50))[-top_n:]
# min_idx = np.argsort(np.abs(imdb-imdb50))[:top_n]
# idx = np.sort(np.array([*min_idx, *max_idx], dtype=int))
idx = np.sort(idx)
idx_queries = imdb[idx].index

imdb_diff = imdb - imdb50

fig, ax = plt.subplots(1, figsize=(7, 3.2))

imdb_faster_idx = idx[np.where((imdb_diff[idx] > 1) | ((imdb_diff[idx] < 0) & (imdb_diff[idx] > -1)))[0]]
imdb50_faster_idx = idx[np.where((imdb_diff[idx] < -1) | ((imdb_diff[idx] > 0) & (imdb_diff[idx] < 1)))[0]]

ci = 1.96 * np.sqrt(imdb_var / 4 + imdb50_var / 4)
ci_data = np.array([0 for _ in ci])


def preprocess_values(x):
    x = np.abs(x)
    return x


imdb_faster_data = [preprocess_values(imdb_diff[x]) if x in imdb_faster_idx else 0 for x in idx]
imdb_50_faster_data = [preprocess_values(imdb_diff[x]) if x in imdb50_faster_idx else 0 for x in idx]

ax.bar(imdb.index[idx], imdb_faster_data, color='tab:blue', label='Better Ratio on IMDB-50%')
ax.bar(imdb.index[idx], imdb_50_faster_data, color='tab:orange', label='Better Ratio on Full IMDB')
ax.bar(imdb.index[idx], ci_data[idx], yerr=ci[idx], label=None, capsize=4)
ax.set_xlim([-0.5, len(idx) - 0.5])
ax.set_ylim([0.1, 100])
ax.set_yscale('log')
ax.legend(loc='upper left')
ax.set_xticklabels(imdb.index[idx], rotation=90)
ax.set_ylabel('Speedup Ratio Differences')

plt.tight_layout()
plt.show()
None

# In[22]:


intresting_queries = ['16b', '17a', '19d']

df_imdb = df[(df['method'] == 'Bao') & (df['experiment'] == 'base_query_split_1')]
display(df_imdb[df_imdb['query_ident'].isin(intresting_queries)])

df_imdb50 = df[(df['method'] == 'Bao') & (df['experiment'] == 'imdb50_trained__imdb_tested')]
display(df_imdb50[df_imdb50['query_ident'].isin(intresting_queries)])

comparison_plot(df_imdb, df_imdb50, threshold=500,
                legend_handles=['Bao Trained on IMDB is Faster', 'Bao Trained on IMDB-50% is Faster'],
                show_overview=False, ymax_override=21)

# In[23]:


df_combined = create_df(df_imdb, df_imdb50, threshold=500)
df_combined[df_combined['significant']].sort_values(['%diff_nobit'])

# # Inference + Planning vs. Training Time

# In[24]:


methods = ['PostgreSQL', 'Neo', 'Bao', 'Balsa', 'LEON', 'HybridQO']
data = {}

for m in methods:
    data[m] = {'train': [], 'test': []}

    for experiment in df_training.index:
        training_time = df_training.loc[experiment, m]

        for split in ['train', 'test']:
            df_selected = df_agg[
                (df_agg['method'] == m) & (df_agg['experiment'] == experiment) & (df_agg['split'] == split)]

            inference_time = df_selected['inference_time'].sum()
            planning_time = df_selected['planning_time'].sum()
            execution_time = df_selected['execution_time'].sum()

            data[m][split].append((training_time, inference_time + planning_time, execution_time))

# In[25]:


fig, ax = plt.subplots(figsize=(4.5, 2.5))

split = 'test'
for m in methods:
    if m not in data.keys():
        continue
    m_data = np.array(data[m][split])

    ax.scatter(m_data[:, 0], (m_data[:, 1] + m_data[:, 2]) / 1000.0, label=m, s=12)
    ax.set_xlabel('End-to-End Training Time [h]')
    ax.set_ylabel('Combined Workload Runtime [s]\n(Inference, Planning\nand Execution Time)', fontsize=9.5)
    ax.legend(loc='upper left', ncol=2, markerscale=2)
    ax.set_yscale('log')
    ax.set_xlim([-3, 128])

plt.show()
None


# ### Direct Comparisons

# In[233]:


def xticks_with_train_split(ax, df, experiment='base_query_split_1'):
    labels = df['query_ident'].apply(lambda x: f"* {x}" if x in TEST_QUERIES[experiment] else x)
    ax.set_xticklabels(labels, rotation=90)


def plot_ax(ax, df, title, ymax, experiment='base_query_split_1', yscale=None):
    ax.set_title(title)
    ax.bar(df['query_ident'], df['execution_time'])
    ax.bar(df['query_ident'], df['planning_time'], bottom=df['execution_time'])
    ax.bar(df['query_ident'], df['inference_time'], bottom=df['execution_time'] + df['planning_time'])

    ax.set_xlim([-0.5, df.shape[0] - 0.5])
    ax.set_ylim(top=ymax * 1.1)
    xticks_with_train_split(ax, df, experiment=experiment)
    ax.set_ylabel('Time [ms]')
    ax.set_xlabel('Query')
    ax.legend(['Execution Time', 'Planning Time', 'Inference Time'])

    if yscale is not None:
        ax.set_yscale(yscale)


def plot_comparison_ax(ax, df, df2, title, ymax, experiment='base_query_split_1', yscale=None):
    ax.set_title(title)
    colors = []
    for x in (df2['total_time'] - df['total_time']):
        if x >= 0:
            colors.append('green')
        else:
            colors.append('red')

    ax.bar(df['query_ident'], df2['total_time'] - df['total_time'], color=colors)
    ax.set_xlim([-0.5, df.shape[0] - 0.5])
    ax.set_ylim([-ymax, ymax])

    xticks_with_train_split(ax, df, experiment=experiment)
    ax.set_ylabel('Difference in Total Time [ms]')
    ax.set_xlabel('Query')

    if yscale is not None:
        ax.set_yscale(yscale)


fig_size = (18, 10)

# ### PostgreSQL vs. Leon

# In[235]:


df_leon = pd.read_csv(os.path.join('leon', 'log__leave_one_out_1.csv'), sep=';')
df_leon['inference_time'] *= 1000.0
df_leon['total_time'] = df_leon['inference_time'] + df_leon['planning_time'] + df_leon['execution_time']

df_leon['query_ident'] = df_leon['query_ident'].apply(lambda x: query_file_to_ident(x))
df_leon = df_leon.sort_values(['query_ident']).reset_index(drop=True)

df_pg = df_agg[(df_agg['experiment'] == 'leave_one_out_split_1') & (df_agg['method'] == 'PostgreSQL')]

fig, axs = plt.subplots(3, figsize=fig_size)
ymax = max([df_pg['total_time'].max(), df_leon['total_time'].max()])

plot_ax(axs[0], df_leon, 'Leon', ymax, yscale='log')
plot_ax(axs[1], df_pg, 'PostgreSQL', ymax, yscale='log')
axs[1].set_ylim(axs[0].get_ylim())
plot_comparison_ax(axs[2], df_leon, df_pg, 'PostgreSQL - Leon', ymax, yscale='symlog')

plt.suptitle('Leon vs. PostgreSQL')
plt.tight_layout()
plt.show()

None

# In[251]:


print(f"\tLeon\t\tPG")
print(f"\t---\t\t---")
counter = 0
for (a, row_leon), (b, row_pg) in zip(df_leon.iterrows(), df_pg.iterrows()):

    leon_time = row_leon['execution_time'] + row_leon['planning_time']
    pg_time = row_pg['execution_time'] + row_pg['planning_time']

    if leon_time + 500 < pg_time:
        print(f"{row_leon['query_ident']}:\t{leon_time:.2f}\t\t{pg_time:.2f}")
        counter += 1
    else:
        continue

print()
print(f"{counter} queries are faster by Leon")

# In[247]:


df_leon = pd.read_csv(os.path.join('leon', 'log__leave_one_out_1.csv'), sep=';')
df_leon['inference_time'] *= 1000.0
df_leon['inference_time'] = 0
df_leon['total_time'] = df_leon['planning_time'] + df_leon['execution_time']

df_leon['query_ident'] = df_leon['query_ident'].apply(lambda x: query_file_to_ident(x))
df_leon = df_leon.sort_values(['query_ident']).reset_index(drop=True)

df_pg = df_agg[(df_agg['experiment'] == 'leave_one_out_split_1') & (df_agg['method'] == 'PostgreSQL')]
df_pg['inference_time'] = 0
df_pg['total_time'] = df_pg['planning_time'] + df_pg['execution_time']

fig, axs = plt.subplots(3, figsize=fig_size)
ymax = max([df_pg['total_time'].max(), df_leon['total_time'].max()])

plot_ax(axs[0], df_leon, 'Leon', ymax)
plot_ax(axs[1], df_pg, 'PostgreSQL', ymax)
axs[1].set_ylim(axs[0].get_ylim())
plot_comparison_ax(axs[2], df_leon, df_pg, 'PostgreSQL - Leon', ymax)

plt.suptitle('Leon vs. PostgreSQL')
plt.tight_layout()
plt.show()

None

# In[ ]:
