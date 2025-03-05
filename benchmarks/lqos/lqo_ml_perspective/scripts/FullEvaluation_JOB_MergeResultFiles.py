#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

# In[3]:


import os
import sys

if os.getcwd().split('/')[-1] == 'notebooks':
    os.chdir('../')
os.getcwd()

# In[4]:


import os
import re
import datetime
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


def query_file_to_ident(file_name):
    ident = file_name.split('.sql')[0]
    return f"{ident[:-1].zfill(2)}{ident[-1]}"


QUERY_TIMEOUT = 2 * 3 * 60 * 1000

# ## Read test queries

# In[6]:


TEST_QUERIES = {
    'base_query_split_1': ['02a', '02b', '02c', '02d', '07a', '07b', '07c', '15a', '15b', '15c', '15d', '24a', '24b',
                           '25a', '25b', '25c', '31a', '31b', '31c'],
    'base_query_split_2': ['13a', '13b', '13c', '13d', '15a', '15b', '15c', '15d', '20a', '20b', '20c', '26a', '26b',
                           '26c', '29a', '29b', '29c', '30a', '30b', '30c', '33a', '33b', '33c'],
    'base_query_split_3': ['01a', '01b', '01c', '01d', '05a', '05b', '05c', '12a', '12b', '12c', '17a', '17b', '17c',
                           '17d', '17e', '17f', '22a', '22b', '22c', '22d', '27a', '27b', '27c', '28a', '28b', '28c'],

    'leave_one_out_split_1': ['01c', '02a', '03b', '04a', '05a', '06b', '07c', '08c', '09c', '10b', '11b', '12c', '13b',
                              '14a', '15b', '16c', '17c', '18b', '19a', '20c', '21c', '22b', '23b', '24a', '25a', '26c',
                              '27c', '28a', '29b', '30a', '31b', '32b', '33c'],
    'leave_one_out_split_2': ['01d', '02d', '03a', '04b', '05c', '06d', '07a', '08c', '09c', '10a', '11a', '12a', '13d',
                              '14b', '15b', '16a', '17f', '18a', '19d', '20a', '21b', '22c', '23b', '24b', '25a', '26a',
                              '27b', '28c', '29a', '30b', '31a', '32b', '33b'],
    'leave_one_out_split_3': ['01c', '02d', '03b', '04a', '05c', '06d', '07b', '08a', '09a', '10c', '11d', '12a', '13a',
                              '14b', '15a', '16d', '17b', '18b', '19d', '20b', '21a', '22a', '23b', '24a', '25b', '26a',
                              '27a', '28b', '29c', '30a', '31a', '32a', '33c'],

    'random_split_1': ['01c', '02c', '04b', '04c', '05c', '06a', '06c', '06e', '08b', '08c', '09c', '11d', '15a', '17b',
                       '17e', '18b', '20a', '21a', '25c', '28b', '32b', '33a'],
    'random_split_2': ['01a', '04c', '05c', '06c', '06d', '07b', '08c', '10a', '11a', '11d', '13c', '13d', '15d', '16a',
                       '17b', '19a', '20a', '22b', '25b', '29b', '31a', '32b'],
    'random_split_3': ['02a', '03b', '06d', '09b', '10b', '11b', '11c', '13c', '13d', '16b', '18c', '19c', '21c', '22a',
                       '22d', '26a', '26b', '27c', '28a', '28c', '30a', '33c'],
}

# ## Read Postgres results

# In[7]:


pg_paths = [
    'test__postgres__job2.txt',

    'test__postgres__job0_m2.txt',
    'test__postgres__job0_m3.txt',
    'test__postgres__job0_m4.txt',
    'test__postgres__job0_m5.txt',

    'test__postgres__job1_m2.txt',
    'test__postgres__job1_m3.txt',
    'test__postgres__job1_m4.txt',
    'test__postgres__job1_m5.txt',

    'test__postgres__job3_m2.txt',
    'test__postgres__job3_m3.txt',
    'test__postgres__job3_m4.txt',
    'test__postgres__job3_m5.txt',

    'test__postgres__job__no_bitmap_tidscan.txt',
    'test__postgres__job__no_bitmap_tidscan2.txt',

    'test__postgres__job__no_geqo__m2.txt',
    'test__postgres__job__no_geqo__m3.txt',
    'test__postgres__job__no_geqo__m4.txt',
    'test__postgres__job__no_geqo__m5.txt',

    'test__postgres__job__no_geqo2__m2.txt',
    'test__postgres__job__no_geqo2__m3.txt',
    'test__postgres__job__no_geqo2__m4.txt',
    'test__postgres__job__no_geqo2__m5.txt',

    'test__postgres__imdb2_tested__0.txt',
    'test__postgres__imdb2_tested__1.txt',
    'test__postgres__imdb2_tested__2.txt',
    'test__postgres__imdb2_tested__3.txt',
]


def read_pg_log(file_name, i):
    pg_data = []

    path = os.path.join('bao', 'logs', file_name)
    with open(path, 'r') as f:
        lines = f.readlines()

    if 'no_geqo' in file_name:
        run_type = 'no_geqo'
    elif 'no_bitmap_tidscan' in file_name:
        run_type = 'no_bitmap_tidscan'
    elif 'imdb2_tested' in file_name:
        run_type = 'imdb50'
    else:
        run_type = 'regular'

    for line in lines:
        hint, iteration, timestamp, query_path, planning, execution, method = line.split('\n')[0].split(', ')

        tmp = {
            'query_ident': query_file_to_ident(query_path.split('/')[-1]),
            'iteration': int(iteration),
            'planning_time': float(planning),
            'execution_time': float(execution),
            'method': 'PostgreSQL',
            'run_id': i,
            'run_type': run_type,
            'timed_out': False,
            'experiment': None,
        }
        tmp['split'] = None
        pg_data.append(tmp)

    df_pg = pd.DataFrame(pg_data)
    df_pg['inference_time'] = 0.0
    df_pg = df_pg[df_pg['iteration'] == 2].reset_index(drop=True)
    df_pg = df_pg.drop(columns=['iteration'])
    df_pg = df_pg[
        ['run_id', 'run_type', 'experiment', 'method', 'query_ident', 'split', 'inference_time', 'planning_time',
         'execution_time', 'timed_out']]
    df_pg = df_pg.sort_values(['query_ident']).reset_index(drop=True)
    return df_pg


all_pg_df = []
cmp_data = dict()
for i, p in enumerate(pg_paths):
    df_tmp = read_pg_log(p, i)
    print(
        f"[{i}] Planning Time: {df_tmp.sum(numeric_only=True)['planning_time']:.2f}\tExecution Time: {df_tmp.sum(numeric_only=True)['execution_time']:.2f}\t{p}")
    cmp_data[i] = df_tmp['execution_time'].tolist()
    # display(df_tmp)
    all_pg_df.append(df_tmp)

df_pg_all = pd.concat(all_pg_df)
display(df_pg_all[df_pg_all['query_ident'] == '30a'])

# ## Read Bao results (based on Bao codebase)

# In[13]:


# only select bao test nd for imdb
print("Get all path of the bao")
for path in sorted(os.listdir(os.path.join('bao', 'logs'))):
    if 'bao' not in path:
        continue

    if 'test' not in path:
        continue

    if 'with_hint' in path:
        continue

    if 'stack' in path:
        continue

    if 'lero' in path:
        continue

    if 'random_walk' in path:
        continue

    print(path)

bao_data = []
# only select bao test nd for imdb
for path in sorted(os.listdir(os.path.join('bao', 'logs'))):
    if 'bao' not in path:
        continue

    if 'test' not in path:
        continue

    if 'with_hint' in path:
        continue

    if 'stack' in path:
        continue

    if 'lero' in path:
        continue

    if 'random_walk' in path:
        continue

    with open(os.path.join('bao', 'logs', path), 'r') as f:
        lines = f.readlines()

    # this is to test performance under distributed shifts
    if 'imdb2' in path:
        path_no_ext = path.split('.txt')[0]
        run_id = path_no_ext[-1]

        experiment = '__'.join(path_no_ext.split('__')[2:4])
        experiment = experiment.replace('imdb2', 'imdb50')

    else:
        num = path.split('.txt')[0][-3:]
        run_id = (int(num[-1]) - 1) if num[:2] == '__' else 0

        # Pattern to find the train test split 
        pattern = re.compile('([a-z]+\_)+split\_[1-9]')
        experiment = pattern.search(path).group()

    for line in lines:
        hint, iteration, timestamp, query_path, planning, execution, method = line.split('\n')[0].split(', ')

        tmp = {
            'query_ident': query_file_to_ident(query_path.split('/')[-1]),
            'iteration': int(iteration),
            'planning_time': float(planning),
            'execution_time': float(execution),
            'method': 'Bao',
            'experiment': experiment,
            'timed_out': False if float(execution) < QUERY_TIMEOUT else True,
            'run_id': run_id,
            'run_type': 'regular',
        }
        experiment_key = tmp['experiment'] if 'imdb2' not in path else 'base_query_split_1'
        tmp['split'] = 'test' if tmp['query_ident'] in TEST_QUERIES[experiment_key] else 'train'

        if tmp['split'] == "train":
            print("debug")

        bao_data.append(tmp)

df_bao = pd.DataFrame(bao_data)
df_bao['inference_time'] = 0.0
df_bao = df_bao[df_bao['iteration'] == 2].reset_index(drop=True)
df_bao = df_bao.drop(columns=['iteration'])
df_bao = df_bao[
    ['run_id', 'run_type', 'experiment', 'method', 'query_ident', 'split', 'inference_time', 'planning_time',
     'execution_time', 'timed_out']]

df_bao[df_bao['query_ident'] == '29a']

# ## Read Neo + Balsa results (based on Balsa codebase)

# In[17]:


file_paths = {
    'neo': {
        0: {
            'base_query_split_1': 'balsa/logs/2023_07_17__115828_plan_and_execute.txt',
            'base_query_split_2': 'balsa/logs/2023_07_24__072811_plan_and_execute.txt',
            'base_query_split_3': 'balsa/logs/2023_07_24__073716_plan_and_execute.txt',
            'leave_one_out_split_1': 'balsa/logs/2023_07_24__071655_plan_and_execute.txt',
            'leave_one_out_split_2': 'balsa/logs/2023_07_24__073722_plan_and_execute.txt',
            'leave_one_out_split_3': 'balsa/logs/2023_07_24__074557_plan_and_execute.txt',
            'random_split_1': 'balsa/logs/2023_07_24__071651_plan_and_execute.txt',
            'random_split_2': 'balsa/logs/2023_07_24__072740_plan_and_execute.txt',
            'random_split_3': 'balsa/logs/2023_07_24__072756_plan_and_execute.txt',
        },
        1: {
            'base_query_split_1': 'balsa/logs/2023_08_03__164448__Neo_JOBBaseQuerySplit1__plan_and_execute.txt',
            'base_query_split_2': 'balsa/logs/2023_08_03__165715__Neo_JOBBaseQuerySplit2__plan_and_execute.txt',
            'base_query_split_3': 'balsa/logs/2023_08_03__170333__Neo_JOBBaseQuerySplit3__plan_and_execute.txt',
            'leave_one_out_split_1': 'balsa/logs/2023_08_03__174147__Neo_JOBLeaveOneOutSplit1__plan_and_execute.txt',
            'leave_one_out_split_2': 'balsa/logs/2023_08_03__174151__Neo_JOBLeaveOneOutSplit2__plan_and_execute.txt',
            'leave_one_out_split_3': 'balsa/logs/2023_08_03__175441__Neo_JOBLeaveOneOutSplit3__plan_and_execute.txt',
            'random_split_1': 'balsa/logs/2023_08_03__163153__Neo_JOBRandomSplit1__plan_and_execute.txt',
            'random_split_2': 'balsa/logs/2023_08_03__163137__Neo_JOBRandomSplit2__plan_and_execute.txt',
            'random_split_3': 'balsa/logs/2023_08_03__164440__Neo_JOBRandomSplit3__plan_and_execute.txt',
        },
        2: {
            'base_query_split_1': 'balsa/logs/2023_08_03__164453__Neo_JOBBaseQuerySplit1__plan_and_execute.txt',
            'base_query_split_2': 'balsa/logs/2023_08_03__174135__Neo_JOBBaseQuerySplit2__plan_and_execute.txt',
            'base_query_split_3': 'balsa/logs/2023_08_03__170341__Neo_JOBBaseQuerySplit3__plan_and_execute.txt',
            'leave_one_out_split_1': 'balsa/logs/2023_08_03__180820__Neo_JOBLeaveOneOutSplit1__plan_and_execute.txt',
            'leave_one_out_split_2': 'balsa/logs/2023_08_03__174211__Neo_JOBLeaveOneOutSplit2__plan_and_execute.txt',
            'leave_one_out_split_3': 'balsa/logs/2023_08_03__182129__Neo_JOBLeaveOneOutSplit3__plan_and_execute.txt',
            'random_split_1': 'balsa/logs/2023_08_03__163348__Neo_JOBRandomSplit1__plan_and_execute.txt',
            'random_split_2': 'balsa/logs/2023_08_03__163145__Neo_JOBRandomSplit2__plan_and_execute.txt',
            'random_split_3': 'balsa/logs/2023_08_03__170316__Neo_JOBRandomSplit3__plan_and_execute.txt',
        }
    },
    'balsa': {
        0: {
            'base_query_split_1': 'balsa/logs/2023_07_13__132703_plan_and_execute.txt',
            'base_query_split_2': 'balsa/logs/2023_07_24__075458_plan_and_execute.txt',
            'base_query_split_3': 'balsa/logs/2023_07_24__080007_plan_and_execute.txt',
            'leave_one_out_split_1': 'balsa/logs/2023_07_24__071715_plan_and_execute.txt',
            'leave_one_out_split_2': 'balsa/logs/2023_07_24__074916_plan_and_execute.txt',
            'leave_one_out_split_3': 'balsa/logs/2023_07_24__080159_plan_and_execute.txt',
            'random_split_1': 'balsa/logs/2023_07_24__071700_plan_and_execute.txt',
            'random_split_2': 'balsa/logs/2023_07_24__074825_plan_and_execute.txt',
            'random_split_3': 'balsa/logs/2023_07_24__075015_plan_and_execute.txt',
        },
        1: {
            'base_query_split_1': 'balsa/logs/2023_08_03__180828__Balsa_JOBLeakageTest2__plan_and_execute.txt',
            'base_query_split_2': 'balsa/logs/2023_08_03__183750__Balsa_JOBBaseQuerySplit2__plan_and_execute.txt',
            'base_query_split_3': 'balsa/logs/2023_08_03__185000__Balsa_JOBBaseQuerySplit3__plan_and_execute.txt',
            'leave_one_out_split_1': 'balsa/logs/2023_08_03__185013__Balsa_JOBLeaveOneOutSplit1__plan_and_execute.txt',
            'leave_one_out_split_2': 'balsa/logs/2023_08_03__191032__Balsa_JOBLeaveOneOutSplit2__plan_and_execute.txt',
            'leave_one_out_split_3': 'balsa/logs/2023_08_03__191037__Balsa_JOBLeaveOneOutSplit3__plan_and_execute.txt',
            'random_split_1': 'balsa/logs/2023_08_03__180835__Balsa_JOBRandomSplit1__plan_and_execute.txt',
            'random_split_2': 'balsa/logs/2023_08_03__175509__Balsa_JOBRandomSplit2__plan_and_execute.txt',
            'random_split_3': 'balsa/logs/2023_08_03__182137__Balsa_JOBRandomSplit3__plan_and_execute.txt',
        },
        2: {
            'base_query_split_1': 'balsa/logs/2023_08_03__180841__Balsa_JOBLeakageTest2__plan_and_execute.txt',
            'base_query_split_2': 'balsa/logs/2023_08_03__190259__Balsa_JOBBaseQuerySplit2__plan_and_execute.txt',
            'base_query_split_3': 'balsa/logs/2023_08_03__185007__Balsa_JOBBaseQuerySplit3__plan_and_execute.txt',
            'leave_one_out_split_1': 'balsa/logs/2023_08_03__191402__Balsa_JOBLeaveOneOutSplit1__plan_and_execute.txt',
            'leave_one_out_split_2': 'balsa/logs/2023_08_03__191922__Balsa_JOBLeaveOneOutSplit2__plan_and_execute.txt',
            'leave_one_out_split_3': 'balsa/logs/2023_08_03__192842__Balsa_JOBLeaveOneOutSplit3__plan_and_execute.txt',
            'random_split_1': 'balsa/logs/2023_08_03__183743__Balsa_JOBRandomSplit1__plan_and_execute.txt',
            'random_split_2': 'balsa/logs/2023_08_03__175520__Balsa_JOBRandomSplit2__plan_and_execute.txt',
            'random_split_3': 'balsa/logs/2023_08_03__184954__Balsa_JOBRandomSplit3__plan_and_execute.txt',
        }
    }
}

# In[18]:


df_neo_balsa = None
for method in file_paths.keys():
    for run_id in file_paths[method].keys():

        method_paths = file_paths[method][run_id]
        for experiment, file_path in method_paths.items():

            df_tmp = pd.read_csv(file_path, header=None, sep=';')
            df_tmp.columns = ['query_ident', 'inference_time', 'planning_time', 'execution_time']
            df_tmp['query_ident'] = df_tmp['query_ident'].apply(lambda x: query_file_to_ident(x))
            df_tmp['method'] = 'Neo' if method == 'neo' else 'Balsa'
            df_tmp['experiment'] = experiment

            df_tmp['timed_out'] = df_tmp['execution_time'].apply(lambda x: True if x < 0 else False)
            df_tmp['run_id'] = int(run_id)
            df_tmp['run_type'] = 'regular'

            df_tmp.loc[df_tmp['planning_time'] == -1, 'planning_time'] = 0.0
            df_tmp.loc[df_tmp['execution_time'] == -1, 'execution_time'] = QUERY_TIMEOUT
            df_tmp['inference_time'] *= 1000.0

            # if len(df_tmp[df_tmp['timed_out']]) > 0:
            #    display(df_tmp[df_tmp['timed_out']])

            df_tmp['split'] = df_tmp['query_ident'].apply(
                lambda x: 'test' if x in TEST_QUERIES[experiment] else 'train')

            if df_neo_balsa is None:
                df_neo_balsa = df_tmp
            else:
                df_neo_balsa = pd.concat([df_neo_balsa, df_tmp]).reset_index(drop=True)
df_neo_balsa = df_neo_balsa[
    ['run_id', 'run_type', 'experiment', 'method', 'query_ident', 'split', 'inference_time', 'planning_time',
     'execution_time', 'timed_out']]

display(df_neo_balsa)

# ## Read LEON results

# In[19]:


leon_files = [
    'log__base_query_split_1.csv',
    'log__base_query_split_2.csv',
    'log__base_query_split_3.csv',
    'log__leave_one_out_1.csv',
    'log__leave_one_out_2.csv',
    'log__leave_one_out_3.csv',
    'log__random_split_1.csv',
    'log__random_split_2.csv',
    'log__random_split_3.csv'
]

dfs = []
for file_name in leon_files:
    file_path = os.path.join('leon', file_name)

    df_tmp = pd.read_csv(file_path, sep=';')
    df_tmp['query_ident'] = df_tmp['query_ident'].apply(query_file_to_ident)
    experiment = file_name.split('__')[1].split('.csv')[0]
    if 'leave_one_out' in experiment:
        experiment = experiment.replace('leave_one_out', 'leave_one_out_split')
    df_tmp['experiment'] = experiment
    df_tmp['run_type'] = 'regular'
    df_tmp['run_id'] = 0
    df_tmp['method'] = 'LEON'

    df_tmp['timed_out'] = False
    df_tmp.loc[df_tmp[df_tmp['execution_time'] > QUERY_TIMEOUT].index, 'timed_out'] = True
    df_tmp.loc[df_tmp[df_tmp['execution_time'] > QUERY_TIMEOUT].index, 'execution_time'] = QUERY_TIMEOUT
    df_tmp['inference_time'] *= 1000.0

    dfs.append(df_tmp)

df_leon = pd.concat(dfs).sort_values(['experiment', 'query_ident', 'run_id']).reset_index(drop=True)
df_leon

# ## Read HybridQO results

# In[66]:


hybridqo_paths = []
for f in sorted(os.listdir('hybrid_qo/logs/wandb_export')):
    if 'JOB' in f:
        hybridqo_paths.append(f)

# Prepare query identifiers from workloads
hybridqo_query_idents = {
    'random_split_1': [],
    'random_split_2': [],
    'random_split_3': [],
    'base_query_split_1': [],
    'base_query_split_2': [],
    'base_query_split_3': [],
    'leave_one_out_split_1': [],
    'leave_one_out_split_2': [],
    'leave_one_out_split_3': []
}

for workload in hybridqo_query_idents.keys():
    workload_path = f'hybrid_qo/workload/JOB__{workload}__'

    train_path = workload_path + 'train.json'
    test_path = workload_path + 'test.json'

    with open(train_path) as f:
        tmp_workload = json.load(f)
    hybridqo_query_idents[workload].extend([line[-2].replace('.sql', '') for line in tmp_workload])

    with open(test_path) as f:
        tmp_workload = json.load(f)
    hybridqo_query_idents[workload].extend([line[-2].replace('.sql', '') for line in tmp_workload])

dfs = []
for file_name in hybridqo_paths:
    file_path = os.path.join('hybrid_qo', 'logs', 'wandb_export', file_name)

    df_tmp = pd.read_csv(file_path)
    df_tmp.columns

    known_column_names = ['hinter_latency', 'test_query', 'epoch', 'mcts_time', 'hinter_plan_time', 'MPHE_time']

    new_columns = []
    select_columns = []
    for c in df_tmp.columns:
        for known_col in known_column_names:
            found = False
            if c.endswith(known_col):
                new_columns.append(known_col)
                select_columns.append(known_col)
                found = True
                break
        if not found:
            new_columns.append(c)

    df_tmp.columns = new_columns
    df_tmp = df_tmp[['Step'] + select_columns]

    # Choose <= epoch 50
    df_epoch = df_tmp.groupby(['epoch']).count()['Step'].to_frame().loc[:50]
    chosen_epoch = df_epoch[df_epoch['Step'] == 113].index.max()
    df_tmp = df_tmp[df_tmp['epoch'] == chosen_epoch]
    df_tmp['execution_time'] = 1000.0 * df_tmp['hinter_latency']
    df_tmp['planning_time'] = 1000.0 * df_tmp['hinter_plan_time']
    df_tmp['inference_time'] = 1000.0 * (df_tmp['mcts_time'] + df_tmp['MPHE_time'])

    df_tmp['timed_out'] = df_tmp['execution_time'] >= QUERY_TIMEOUT

    file_name = file_name.split('.csv')[0]
    method, workload, experiment, run_id = file_name.split('__')

    df_tmp['run_id'] = int(run_id)
    df_tmp['run_type'] = 'regular'
    df_tmp['experiment'] = experiment
    df_tmp['method'] = 'HybridQO'
    df_tmp['query_ident'] = hybridqo_query_idents[experiment]
    df_tmp['split'] = df_tmp['test_query'].apply(lambda x: 'train' if x == 0 else 'test')

    df_tmp = df_tmp[
        ['run_id', 'run_type', 'experiment', 'method', 'query_ident', 'split', 'inference_time', 'planning_time',
         'execution_time', 'timed_out']]
    dfs.append(df_tmp)

if len(dfs) > 0:
    df_hybridqo = pd.concat(dfs).sort_values(['experiment', 'query_ident', 'run_id']).reset_index(drop=True)
    display(df_hybridqo)
else:
    None

# ## Combine all results into a single dataframe

# In[67]:


df = pd.concat([df_pg_all, df_bao, df_neo_balsa, df_leon, df_hybridqo]).sort_values(
    ['experiment', 'method', 'query_ident']).reset_index(drop=True)
df['total_time'] = df['inference_time'] + df['planning_time'] + df['execution_time']
df = df[['run_id', 'run_type', 'experiment', 'method', 'query_ident', 'split', 'inference_time', 'planning_time',
         'execution_time', 'total_time', 'timed_out']]
df

# In[70]:


df.groupby(['run_type', 'experiment', 'method', 'timed_out'], dropna=False).count()

# In[71]:


# new_file_name = 'experiment_logs/' + datetime.datetime.now().strftime('%Y%m%d') + '__combined.csv'
# df.to_csv(new_file_name, index=False)
#
# print(f"Saved combined results to {new_file_name}")

# In[ ]:
