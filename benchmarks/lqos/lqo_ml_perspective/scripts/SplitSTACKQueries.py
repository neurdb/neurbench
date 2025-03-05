#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np

if os.getcwd().split('/')[-1] == 'notebooks':
    os.chdir('../')
os.getcwd()

queries = {}

for file in os.listdir('bao/queries/stack'):
    base_query, query_name = file.split('__')

    # remove query9 and query10
    if base_query in ['q9', 'q10']:
        continue

    if base_query not in queries.keys():
        queries[base_query] = list()

    queries[base_query].append((base_query, query_name))

all_queries = []
for base_query in sorted(queries.keys()):
    print('__'.join(queries[base_query][0]))

    for x in queries[base_query]:
        all_queries.append(x)

print(f"# Queries: {len(all_queries)}")

# In[25]:


# for q_dir, q_file in all_queries:
#     file_path = f'data/stack/so_queries/{q_dir}/{q_file.replace(".sql.sql", ".sql")}'
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#     sql = ''.join(lines)
#
#     query_ident = '__'.join([file_path.split('/')[-2], file_path.split('/')[-1]])
#
#     os.makedirs('bao/queries/stack', exist_ok=True)
#     file_path = f"bao/queries/stack/{query_ident}"
#     with open(file_path, 'w') as f:
#         f.write(sql)
#
#     os.makedirs('balsa/queries/stack', exist_ok=True)
#     file_path = f"balsa/queries/stack/{query_ident}"
#     with open(file_path, 'w') as f:
#         f.write(sql)

# ### Random Sampling

# In[27]:


tmp = []

for q_dir, q_ident in all_queries:
    file_path = f'data/stack/so_queries/{q_dir}/{q_ident.replace(".sql.sql", ".sql")}'
    with open(file_path, 'r') as f:
        lines = f.readlines()
    sql = ''.join(lines)

    tmp.append({
        'query_ident': q_ident,
        'query_template': q_dir,
        'full_query_file_name': f"{q_dir}__{q_ident}",
        'sql': sql,
        'file_path': file_path
    })

df = pd.DataFrame(tmp).sort_values(['query_template', 'query_ident']).reset_index(drop=True)
df


# In[32]:


def save_queries(df: pd.DataFrame, base_target_dir: str, test: bool = False):
    target_dir = os.path.join(base_target_dir, ('test' if test else 'train'))
    os.makedirs(target_dir, exist_ok=True)

    for i, row in df.iterrows():
        sql = row['sql']
        file_name = row['full_query_file_name']

        file_path = os.path.join(target_dir, file_name)
        # with open(file_path, 'w') as f:
        #     f.writelines(sql)

    if test:
        print()
        print(target_dir)
        print(sorted(df['full_query_file_name'].tolist()))


# In[33]:


n_test__random = int(df.shape[0] * 0.2) + 1

random_test_1 = df.sample(n=n_test__random, replace=False, random_state=10).sort_index()
random_train_1 = df[~df.index.isin(random_test_1.index)]
save_queries(random_train_1, 'bao/queries/stack__random_split_1', test=False)
save_queries(random_test_1, 'bao/queries/stack__random_split_1', test=True)

"""
random_test_2 = df.sample(n=n_test__random, replace=False, random_state=20).sort_index()
random_train_2 = df[~df.index.isin(random_test_2.index)]
save_queries(random_train_2, 'bao/queries/stack__random_split_2', test=False)
save_queries(random_test_2, 'bao/queries/stack__random_split_2', test=True)

random_test_3 = df.sample(n=n_test__random, replace=False, random_state=30).sort_index()
random_train_3 = df[~df.index.isin(random_test_3.index)]
save_queries(random_train_3, 'bao/queries/stack__random_split_3', test=False)
save_queries(random_test_3, 'bao/queries/stack__random_split_3', test=True)
"""


# ### Leave One Out Sampling

# In[34]:


def split_loo(df, random_state):
    test_idx = []
    for query_template, df_group in df.groupby(['query_template']):
        test_idx.append(df_group.sample(n=1, replace=False, random_state=random_state).iloc[0].name)

    return df[~df.index.isin(test_idx)], df[df.index.isin(test_idx)]


loo1_train, loo1_test = split_loo(df, random_state=10)
save_queries(loo1_train, 'bao/queries/stack__leave_one_out_split_1', test=False)
save_queries(loo1_test, 'bao/queries/stack__leave_one_out_split_1', test=True)

"""
loo2_train, loo2_test = split_loo(df, random_state=20)
save_queries(loo2_train, 'bao/queries/stack__leave_one_out_split_2', test=False)
save_queries(loo2_test, 'bao/queries/stack__leave_one_out_split_2', test=True)

loo3_train, loo3_test = split_loo(df, random_state=30)
save_queries(loo3_train, 'bao/queries/stack__leave_one_out_split_3', test=False)
save_queries(loo3_test, 'bao/queries/stack__leave_one_out_split_3', test=True)
"""


# ### Base Query Sampling

# In[35]:


# 16 query templates @ STACK -> 13-3 ~ 80-20 split
def split_base_query(df, n_test_templates=3):
    test_templates = np.random.choice(df['query_template'].unique(), size=n_test_templates, replace=False)
    return df[~df['query_template'].isin(test_templates)], df[df['query_template'].isin(test_templates)]


np.random.seed(10)
base_query1_train, base_query1_test = split_base_query(df)
save_queries(base_query1_train, 'bao/queries/stack__base_query_split_1', test=False)
save_queries(base_query1_test, 'bao/queries/stack__base_query_split_1', test=True)

"""
np.random.seed(20)
base_query2_train, base_query2_test = split_base_query(df)
save_queries(base_query2_train, 'bao/queries/stack__base_query_split_2', test=False)
save_queries(base_query2_test, 'bao/queries/stack__base_query_split_2', test=True)

np.random.seed(30)
base_query3_train, base_query3_test = split_base_query(df)
save_queries(base_query3_train, 'bao/queries/stack__base_query_split_3', test=False)
save_queries(base_query3_test, 'bao/queries/stack__base_query_split_3', test=True)
"""

# In[ ]:
