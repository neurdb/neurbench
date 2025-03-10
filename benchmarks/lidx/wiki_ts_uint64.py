# %%
# -*- coding: utf-8 -*-
# @ Desc: Example to process wiki_ts dataset

import os

workspace_path = '../..'
os.chdir(workspace_path)
print("Current workspace:", os.getcwd())

import numpy as np

from neurbench.drift import find_q
from neurbench.index.bench import KeySetBinEncoder, sample_bulkloading_keyset
from neurbench.index.util import KeyType, load_key_set, save_file

# %%
# load wiki_ts_200M key set
# keyset_file_path = "/users/lingze/TLI/data/wiki_ts_200M_uint64"
# data = load_key_set(keyset_file_path)

# %%
# get data type and check whether unique
# data_type = KeyType.resolve_type_from_filename(keyset_file_path)
# print(data_type)
# flag = len(np.unique(data)) == len(data)
# print("Data is unique:", flag)

# %%
# save the unique data in the keys
# unique_data = np.unique(data)
# keyset_output_file_path = "/users/lingze/TLI/data/wiki_ts_200M_unique_uint64"
# save_file(keyset_output_file_path, unique_data, data_type)

# %%
# load wiki_ts_unique_200M_uint64 key set
keyset_file_path = "/users/lingze/TLI/data/wiki_ts_200M_unique_uint64"
data = load_key_set(keyset_file_path)

# %%
# get data type and check whether unique
data_type = KeyType.resolve_type_from_filename(keyset_file_path)
print(data_type)
flag = len(np.unique(data)) == len(data)
print("Data is unique:", flag)

# %%
"""plot the data Cumulative Distribution Function (CDF)
"""
max_value = np.max(data)
min_value = np.min(data)
print("Max value:", max_value)
print("Min value:", min_value)
KeySetBinEncoder.plot_cdf(
    data,
    min_value=min_value,
    max_value=max_value,
    offset= 1e7,
)

# %%
"""abnormal key filter
# before we get the data distribution of key set
# we need to filter out the abnormal keys
"""
keys, removed_key = KeySetBinEncoder.filter_abnormal_values(data, "IQR", verbose = True)
print("Remove rate:", len(removed_key) / len(data))

# %%
"""Bin the key set, get the distribution of key set
heuristically choose the bin size.
"""
bin_width_offset = 20
bin_idxs, prob, bin_idx_to_keys = KeySetBinEncoder.bin_keyset_to_distribution(
    keys,
    bin_width_offset=bin_width_offset,
    verbose=True
)
KeySetBinEncoder.plot_probability_distribution(prob, bin_idxs.tolist())

# %%
min_bin_num = int(np.min(bin_idxs))
max_bin_num = int(np.max(bin_idxs))
prob_dict = {}

for i, idx in enumerate(bin_idxs):
    prob_dict[idx] = prob[i]

x = [ i for i in range(min_bin_num, max_bin_num+1)]
y = [ prob_dict[i] if i in prob_dict else 0.0 for i in range(min_bin_num, max_bin_num+1) ]

# %%
# draw the estimated distribution and store the figure
import matplotlib.pyplot as plt

labels = x
max_value = max(y)
rounded_max_value = np.ceil(max_value * 50) / 50
# rounded 0.05
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.bar(labels, y, color='blue', alpha=0.7)
ax.set_xlabel('Bin Number', fontsize = 15)
ax.set_ylabel('Prob', fontsize = 15)
ax.set_title('Empirical Probability Distribution', fontsize = 15)
print(rounded_max_value)
ax.set_ylim(0, 0.02)  # Set y-axis limits from 0 to 1
ax.grid(axis='y')
fig.tight_layout()
filepath = "/users/lingze/neurbench/data/keys/wiki_ts_200M_uint64_distribution.pdf"
fig.savefig(filepath, dpi = 800, bbox_inches = 'tight', pad_inches=0)
# Show the plot
fig.show()

# %% [markdown]
# ---
# ### Drift 0.1

# %%
# Drift setting alpha = 0.1, unform drift
init_prob = find_q(prob, 0.1, skewed=False)
KeySetBinEncoder.plot_probability_distribution(init_prob)

# %%
bulkloading_n = int(0.5 * len(keys))
bulkloading_keys = sample_bulkloading_keyset(
    bin_idxs,
    init_prob,
    bin_idx_to_keys,
    bulkloading_n,
    verbose=True
)
print(len(bulkloading_keys)) 
print("Bulkloading key set size:", len(bulkloading_keys) / len(keys))

# %%
"""Bin the key set, get the distribution of key set
heuristically choose the bin size.
"""
bin_width_offset = 20
bin_idxs, ini_prob, _ = KeySetBinEncoder.bin_keyset_to_distribution(
    np.array(bulkloading_keys),
    bin_width_offset=bin_width_offset,
    verbose=True
)
KeySetBinEncoder.plot_probability_distribution(ini_prob, bin_idxs.tolist())

# %%
inserted_keys = np.setdiff1d(data, bulkloading_keys)
print("Insert number of records:", len(inserted_keys))
print("Insert ratio:", len(inserted_keys) / len(data))

# %%

# draw the picture of the bulkloading key 
prob_dict = {}
for i, idx in enumerate(bin_idxs):
    prob_dict[idx] = ini_prob[i]
y = [ prob_dict[i] if i in prob_dict else 0.0 for i in range(min_bin_num, max_bin_num+1) ]

# %%
# draw the estimated distribution and store the figure
import matplotlib.pyplot as plt

labels = x
max_value = max(y)
rounded_max_value = np.ceil(max_value * 10) / 10
# rounded 0.05
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.bar(labels, y, color='blue', alpha=0.7)
ax.set_xlabel('Bin Number', fontsize = 15)
ax.set_ylabel('Prob', fontsize = 15)
ax.set_title('Drift Probability Distribution', fontsize = 15)
print(rounded_max_value)
ax.set_ylim(0, 0.02)  # Set y-axis limits from 0 to 1
ax.grid(axis='y')
fig.tight_layout()
filepath = "/users/lingze/neurbench/data/keys/wiki_ts_200M_uint64_distribution_drift_01.pdf"
fig.savefig(filepath, dpi = 800, bbox_inches = 'tight', pad_inches=0)
# Show the plot
fig.show()

# %%
workload_dir_path = "/users/lingze/neurbench/data/workload/wiki_ts_200M_uint64_drift_01"
os.makedirs(workload_dir_path, exist_ok=True)

inserted_keys = np.array(inserted_keys, dtype=data_type.to_numpy_type())
bulkloading_keys = np.array(bulkloading_keys, dtype=data_type.to_numpy_type())

bulkloading_keyset_file_path = os.path.join(workload_dir_path, "init_keys_uint64")
insert_keyset_file_path = os.path.join(workload_dir_path, "insert_keys_uint64")
save_file(bulkloading_keyset_file_path, bulkloading_keys, data_type)
save_file(insert_keyset_file_path, inserted_keys, data_type)

# %% [markdown]
# ---
# ### Drift 0.3

# %%
# Drift setting alpha = 0.3, unform drift
init_prob = find_q(prob, 0.3, skewed=False)
KeySetBinEncoder.plot_probability_distribution(init_prob)

# %%
bulkloading_n = int(0.5 * len(keys))
bulkloading_keys = sample_bulkloading_keyset(
    bin_idxs,
    init_prob,
    bin_idx_to_keys,
    bulkloading_n,
    verbose=True
)
print(len(bulkloading_keys)) 
print("Bulkloading key set size:", len(bulkloading_keys) / len(keys))

# %%
"""Bin the key set, get the distribution of key set
heuristically choose the bin size.
"""
bin_width_offset = 20
bin_idxs, ini_prob, _ = KeySetBinEncoder.bin_keyset_to_distribution(
    np.array(bulkloading_keys),
    bin_width_offset=bin_width_offset,
    verbose=True
)
KeySetBinEncoder.plot_probability_distribution(ini_prob, bin_idxs.tolist())

# %%
inserted_keys = np.setdiff1d(data, bulkloading_keys)
print("Insert number of records:", len(inserted_keys))
print("Insert ratio:", len(inserted_keys) / len(data))
# print bulkloading key CDF

# %%

# draw the picture of the bulkloading key 
prob_dict = {}
for i, idx in enumerate(bin_idxs):
    prob_dict[idx] = ini_prob[i]
y = [ prob_dict[i] if i in prob_dict else 0.0 for i in range(min_bin_num, max_bin_num+1) ]

# %%
# draw the estimated distribution and store the figure
import matplotlib.pyplot as plt

labels = x
max_value = max(y)
rounded_max_value = np.ceil(max_value * 10) / 10
# rounded 0.05
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.bar(labels, y, color='blue', alpha=0.7)
ax.set_xlabel('Bin Number', fontsize = 15)
ax.set_ylabel('Prob', fontsize = 15)
ax.set_title('Drift Probability Distribution', fontsize = 15)
print(rounded_max_value)
ax.set_ylim(0, 0.02)  # Set y-axis limits from 0 to 1
ax.grid(axis='y')
fig.tight_layout()
filepath = "/users/lingze/neurbench/data/keys/wiki_ts_200M_uint64_distribution_drift_03.pdf"
fig.savefig(filepath, dpi = 800, bbox_inches = 'tight', pad_inches=0)
# Show the plot
fig.show()

# %%
workload_dir_path = "/users/lingze/neurbench/data/workload/wiki_ts_200M_uint64_drift_03"
os.makedirs(workload_dir_path, exist_ok=True)

inserted_keys = np.array(inserted_keys, dtype=data_type.to_numpy_type())
bulkloading_keys = np.array(bulkloading_keys, dtype=data_type.to_numpy_type())

bulkloading_keyset_file_path = os.path.join(workload_dir_path, "init_keys_uint64")
insert_keyset_file_path = os.path.join(workload_dir_path, "insert_keys_uint64")
save_file(bulkloading_keyset_file_path, bulkloading_keys, data_type)
save_file(insert_keyset_file_path, inserted_keys, data_type)

# %% [markdown]
# ---
# ### Drift 0.5

# %%
# Drift setting alpha = 0.5, unform drift
init_prob = find_q(prob, 0.5, skewed=False)
KeySetBinEncoder.plot_probability_distribution(init_prob)

# %%
bulkloading_n = int(0.7*len(keys))
bulkloading_keys = sample_bulkloading_keyset(
    bin_idxs,
    init_prob,
    bin_idx_to_keys,
    bulkloading_n,
    verbose=True
)
print(len(bulkloading_keys)) 
print("Bulkloading key set size:", len(bulkloading_keys) / len(keys))

# %%
"""Bin the key set, get the distribution of key set
heuristically choose the bin size.
"""
bin_width_offset = 20
bin_idxs, ini_prob, _ = KeySetBinEncoder.bin_keyset_to_distribution(
    np.array(bulkloading_keys),
    bin_width_offset=bin_width_offset,
    verbose=True
)
KeySetBinEncoder.plot_probability_distribution(ini_prob, bin_idxs.tolist())

# %%
inserted_keys = np.setdiff1d(data, bulkloading_keys)
print("Insert number of records:", len(inserted_keys))
print("Insert ratio:", len(inserted_keys) / len(data))
# print bulkloading key CDF

# %%

# draw the picture of the bulkloading key 
prob_dict = {}
for i, idx in enumerate(bin_idxs):
    prob_dict[idx] = ini_prob[i]
y = [ prob_dict[i] if i in prob_dict else 0.0 for i in range(min_bin_num, max_bin_num+1) ]

# %%
# draw the estimated distribution and store the figure
import matplotlib.pyplot as plt

labels = x
max_value = max(y)
rounded_max_value = np.ceil(max_value * 10) / 10
# rounded 0.05
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.bar(labels, y, color='blue', alpha=0.7)
ax.set_xlabel('Bin Number', fontsize = 15)
ax.set_ylabel('Prob', fontsize = 15)
ax.set_title('Drift Probability Distribution', fontsize = 15)
print(rounded_max_value)
ax.set_ylim(0, 0.02)  # Set y-axis limits from 0 to 1
ax.grid(axis='y')
fig.tight_layout()
filepath = "/users/lingze/neurbench/data/keys/wiki_ts_200M_uint64_distribution_drift_05.pdf"
fig.savefig(filepath, dpi = 800, bbox_inches = 'tight', pad_inches=0)
# Show the plot
fig.show()

# %%
workload_dir_path = "/users/lingze/neurbench/data/workload/wiki_ts_200M_uint64_drift_05"
os.makedirs(workload_dir_path, exist_ok=True)

inserted_keys = np.array(inserted_keys, dtype=data_type.to_numpy_type())
bulkloading_keys = np.array(bulkloading_keys, dtype=data_type.to_numpy_type())

bulkloading_keyset_file_path = os.path.join(workload_dir_path, "init_keys_uint64")
insert_keyset_file_path = os.path.join(workload_dir_path, "insert_keys_uint64")
save_file(bulkloading_keyset_file_path, bulkloading_keys, data_type)
save_file(insert_keyset_file_path, inserted_keys, data_type)

# %% [markdown]
# ---
# ### Drift 0.7

# %%
# Drift setting alpha = 0.7, unform drift
init_prob = find_q(prob, 0.7, skewed=False)
KeySetBinEncoder.plot_probability_distribution(init_prob)

# %%
bulkloading_n = int( 1 * len(keys))
bulkloading_keys = sample_bulkloading_keyset(
    bin_idxs,
    init_prob,
    bin_idx_to_keys,
    bulkloading_n,
    verbose=True
)
print(len(bulkloading_keys)) 
print("Bulkloading key set size:", len(bulkloading_keys) / len(keys))

# %%
"""Bin the key set, get the distribution of key set
heuristically choose the bin size.
"""
bin_width_offset = 20
bin_idxs, ini_prob, _ = KeySetBinEncoder.bin_keyset_to_distribution(
    np.array(bulkloading_keys),
    bin_width_offset=bin_width_offset,
    verbose=True
)
KeySetBinEncoder.plot_probability_distribution(ini_prob, bin_idxs.tolist())

# %%
inserted_keys = np.setdiff1d(data, bulkloading_keys)
print("Insert number of records:", len(inserted_keys))
print("Insert ratio:", len(inserted_keys) / len(data))
# print bulkloading key CDF

# %%

# draw the picture of the bulkloading key 
prob_dict = {}
for i, idx in enumerate(bin_idxs):
    prob_dict[idx] = ini_prob[i]
y = [ prob_dict[i] if i in prob_dict else 0.0 for i in range(min_bin_num, max_bin_num+1) ]

# %%
# draw the estimated distribution and store the figure
import matplotlib.pyplot as plt

labels = x
max_value = max(y)
rounded_max_value = np.ceil(max_value * 10) / 10
# rounded 0.05
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.bar(labels, y, color='blue', alpha=0.7)
ax.set_xlabel('Bin Number', fontsize = 15)
ax.set_ylabel('Prob', fontsize = 15)
ax.set_title('Drift Probability Distribution', fontsize = 15)
print(rounded_max_value)
ax.set_ylim(0, 0.02)  # Set y-axis limits from 0 to 1
ax.grid(axis='y')
fig.tight_layout()
filepath = "/users/lingze/neurbench/data/keys/wiki_ts_200M_uint64_distribution_drift_07.pdf"
fig.savefig(filepath, dpi = 800, bbox_inches = 'tight', pad_inches=0)
# Show the plot
fig.show()

# %%
workload_dir_path = "/users/lingze/neurbench/data/workload/wiki_ts_200M_uint64_drift_07"
os.makedirs(workload_dir_path, exist_ok=True)

inserted_keys = np.array(inserted_keys, dtype=data_type.to_numpy_type())
bulkloading_keys = np.array(bulkloading_keys, dtype=data_type.to_numpy_type())

bulkloading_keyset_file_path = os.path.join(workload_dir_path, "init_keys_uint64")
insert_keyset_file_path = os.path.join(workload_dir_path, "insert_keys_uint64")
save_file(bulkloading_keyset_file_path, bulkloading_keys, data_type)
save_file(insert_keyset_file_path, inserted_keys, data_type)

# %%



