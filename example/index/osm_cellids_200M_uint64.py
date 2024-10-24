# %%
# -*- coding: utf-8 -*-
# @ Desc: Example to process the osm_cellids data

# set workspace to the root of the project
import os
workspace_path = '../..'
os.chdir(workspace_path)
print("Current workspace:", os.getcwd())

# %%
import numpy as np
from neuralbench.index.util import load_key_set, KeyType, save_file
from neuralbench.index.bench import KeySetBinEncoder, sample_bulkloading_keyset
from neuralbench.drift import find_q
import matplotlib.pyplot as plt

# %%
import matplotlib.font_manager as font_manager

font_path = "/users/lingze/times.ttf"
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
print(prop.get_name())

# %%
# # load osm_cellids data and downsample it to 200M keys
# keyset_file_path = "/users/lingze/TLI/data/osm_cellids_800M_uint64"
# output_file_path = "/users/lingze/TLI/data/osm_cellids_200M_uint64"
# d = np.fromfile(keyset_file_path, dtype=np.uint64)[1:]

# nd = d[::4]
# with open(output_file_path, "wb") as f:
#     f.write(np.array([nd.size], dtype=np.uint64))
#     nd.tofile(f)


# %%
# load osm_cellids_200M_uint64 data
keyset_file_path = "/users/lingze/TLI/data/osm_cellids_200M_uint64"
data = load_key_set(keyset_file_path)


# %%
# get data type and check unique
data_type = KeyType.resolve_type_from_filename(keyset_file_path)
print(data_type)
# flag = len(np.unique(data)) == len(data)
# print("Data is unique:", flag)

# %%
"""plot the data Cumulative Distribution Function (CDF)
"""
# max_value = np.max(data)
# min_value = np.min(data)
# print("Max value:", max_value)
# print("Min value:", min_value)
# KeySetBinEncoder.plot_cdf(
#     data,
#     min_value=min_value,
#     max_value=max_value,
# )

# %%
"""abnormal key filter
# before we get the data distribution of key set
# we need to filter out the abnormal keys
"""
keys, removed_key = KeySetBinEncoder.filter_abnormal_values(data, "CONFIDENCE", verbose = True)
print("Remove rate:", len(removed_key) / len(data))

# %%
"""Bin the key set, get the distribution of key set
heuristically choose the bin size.
"""
bin_width_offset = 55
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

# labels = x
# max_value = max(y)
# rounded_max_value = np.ceil(max_value * 50) / 50
# # rounded 0.05
# y_ = [ i * 100 for i in y]
# plt.rcParams['xtick.labelsize']=15
# plt.rcParams['ytick.labelsize']=15
# fig, ax = plt.subplots(figsize=(4, 2.5))
# ax.bar(labels, y_, color='#1F77B4', alpha=0.7)
# ax.set_xlabel('#Bin', fontsize = 15, fontname='Times New Roman')
# ax.set_ylabel('Prob (%)', fontsize = 15, fontname='Times New Roman')
# ax.set_ylim(0, 2)  # Set y-axis limits from 0 to 1
# ax.set_xlim(-10, 310)
# ax.grid(axis='y')
# fig.tight_layout()
# filepath = "/users/lingze/neurbench/data/keys/osm_cellids_200M_uint64_distribution.pdf"
# fig.savefig(filepath, dpi = 800, bbox_inches = 'tight', pad_inches=0)
# # Show the plot
# fig.show()

# %%
filepath = "/users/lingze/neurbench/data/draw/osm_cellid_origin.npz"
x = np.array(x)
y = np.array(y)
np.savez(filepath, x=x, y=y)

# %% [markdown]
# ---
# ### Drift 0.1

# %%
# Drift setting alpha = 0.1, unform drift
init_prob = find_q(prob, 0.1, skewed=False)
KeySetBinEncoder.plot_probability_distribution(init_prob)

# %%
bulkloading_n = int(0.52 * len(keys))
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
bin_width_offset = 55
bin_idxs, ini_prob, _ = KeySetBinEncoder.bin_keyset_to_distribution(
    np.array(bulkloading_keys),
    bin_width_offset=bin_width_offset,
    verbose=True
)
# KeySetBinEncoder.plot_probability_distribution(ini_prob, bin_idxs.tolist())

# %%
# inserted_keys = np.setdiff1d(data, bulkloading_keys)
# print("Insert number of records:", len(inserted_keys))
# print("Insert ratio:", len(inserted_keys) / len(data))

# %%

# draw the picture of the bulkloading key 
prob_dict = {}
for i, idx in enumerate(bin_idxs):
    prob_dict[idx] = ini_prob[i]
y = [ prob_dict[i] if i in prob_dict else 0.0 for i in range(min_bin_num, max_bin_num+1) ]

# %%
# draw the estimated distribution and store the figure

# labels = x
# max_value = max(y)
# rounded_max_value = np.ceil(max_value * 50) / 50
# # rounded 0.05
# y_ = [ i * 100 for i in y]
# plt.rcParams['xtick.labelsize']=15
# plt.rcParams['ytick.labelsize']=15
# fig, ax = plt.subplots(figsize=(4, 2.5))
# ax.bar(labels, y_, color='blue', alpha=0.7)
# ax.set_xlabel('#Bin', fontsize = 15, fontname='Times New Roman')
# ax.set_ylabel('Prob (%)', fontsize = 15, fontname='Times New Roman')
# ax.set_ylim(0, 2)  # Set y-axis limits from 0 to 1
# ax.set_xlim(-10, 310)
# ax.grid(axis='y')
# fig.tight_layout()
# filepath = "/users/lingze/neurbench/data/keys/osm_cellids_200M_uint64_distribution_drift_01.pdf"
# fig.savefig(filepath, dpi = 800, bbox_inches = 'tight', pad_inches=0)
# # Show the plot
# fig.show()

# %%
filepath = "/users/lingze/neurbench/data/draw/osm_cellid_drift_01.npz"
x = np.array(x)
y = np.array(y)
np.savez(filepath, x=x, y=y)

# %% [markdown]
# ---
# ### Drift 0.3

# %%
# Drift setting alpha = 0.1, unform drift
init_prob = find_q(prob, 0.3, skewed=False)
KeySetBinEncoder.plot_probability_distribution(init_prob)

# %%
bulkloading_n = int(0.52 * len(keys))
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
bin_width_offset = 55
bin_idxs, ini_prob, _ = KeySetBinEncoder.bin_keyset_to_distribution(
    np.array(bulkloading_keys),
    bin_width_offset=bin_width_offset,
    verbose=True
)
# KeySetBinEncoder.plot_probability_distribution(ini_prob, bin_idxs.tolist())

# %%

# draw the picture of the bulkloading key 
prob_dict = {}
for i, idx in enumerate(bin_idxs):
    prob_dict[idx] = ini_prob[i]
y = [ prob_dict[i] if i in prob_dict else 0.0 for i in range(min_bin_num, max_bin_num+1) ]

# %%
# # draw the estimated distribution and store the figure

# labels = x
# max_value = max(y)
# rounded_max_value = np.ceil(max_value * 50) / 50
# # rounded 0.05
# y_ = [ i * 100 for i in y]
# plt.rcParams['xtick.labelsize']=15
# plt.rcParams['ytick.labelsize']=15
# fig, ax = plt.subplots(figsize=(4, 2.5))
# ax.bar(labels, y_, color='#1F77B4', alpha=0.7)
# ax.set_xlabel('#Bin', fontsize = 15, fontname='Times New Roman')
# ax.set_ylabel('Prob (%)', fontsize = 15, fontname='Times New Roman')
# ax.set_ylim(0, 2)  # Set y-axis limits from 0 to 1
# ax.set_xlim(-10, 310)
# ax.grid(axis='y')
# fig.tight_layout()
# filepath = "/users/lingze/neurbench/data/keys/osm_cellids_200M_uint64_distribution_drift_03.pdf"
# fig.savefig(filepath, dpi = 800, bbox_inches = 'tight', pad_inches=0)
# # Show the plot
# fig.show()

# %%
filepath = "/users/lingze/neurbench/data/draw/osm_cellid_drift_03.npz"
x = np.array(x)
y = np.array(y)
np.savez(filepath, x=x, y=y)

# %% [markdown]
# ---
# ### Drift 0.5

# %%
# Drift setting alpha = 0.1, unform drift
init_prob = find_q(prob, 0.5, skewed=False)
KeySetBinEncoder.plot_probability_distribution(init_prob)

# %%
bulkloading_n = int(0.6 * len(keys))
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
bin_width_offset = 55
bin_idxs, ini_prob, _ = KeySetBinEncoder.bin_keyset_to_distribution(
    np.array(bulkloading_keys),
    bin_width_offset=bin_width_offset,
    verbose=True
)
# KeySetBinEncoder.plot_probability_distribution(ini_prob, bin_idxs.tolist())

# %%

# draw the picture of the bulkloading key 
prob_dict = {}
for i, idx in enumerate(bin_idxs):
    prob_dict[idx] = ini_prob[i]
y = [ prob_dict[i] if i in prob_dict else 0.0 for i in range(min_bin_num, max_bin_num+1) ]

# %%
# # draw the estimated distribution and store the figure

# labels = x
# max_value = max(y)
# rounded_max_value = np.ceil(max_value * 50) / 50
# # rounded 0.05
# y_ = [ i * 100 for i in y]
# plt.rcParams['xtick.labelsize']=15
# plt.rcParams['ytick.labelsize']=15
# fig, ax = plt.subplots(figsize=(4, 2.5))
# ax.bar(labels, y_, color='#1F77B4', alpha=0.7)
# ax.set_xlabel('#Bin', fontsize = 15, fontname='Times New Roman')
# ax.set_ylabel('Prob (%)', fontsize = 15, fontname='Times New Roman')
# ax.set_ylim(0, 2)  # Set y-axis limits from 0 to 1
# ax.set_xlim(-10, 310)
# ax.grid(axis='y')
# fig.tight_layout()
# filepath = "/users/lingze/neurbench/data/keys/osm_cellids_200M_uint64_distribution_drift_05.pdf"
# fig.savefig(filepath, dpi = 800, bbox_inches = 'tight', pad_inches=0)
# # Show the plot
# fig.show()

# %%
filepath = "/users/lingze/neurbench/data/draw/osm_cellid_drift_05.npz"
x = np.array(x)
y = np.array(y)
np.savez(filepath, x=x, y=y)

# %% [markdown]
# ---
# ### Drift 0.7

# %%
# Drift setting alpha = 0.1, unform drift
init_prob = find_q(prob, 0.7, skewed=False)
KeySetBinEncoder.plot_probability_distribution(init_prob)

# %%
bulkloading_n = int(1 * len(keys))
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
bin_width_offset = 55
bin_idxs, ini_prob, _ = KeySetBinEncoder.bin_keyset_to_distribution(
    np.array(bulkloading_keys),
    bin_width_offset=bin_width_offset,
    verbose=True
)
# KeySetBinEncoder.plot_probability_distribution(ini_prob, bin_idxs.tolist())

# %%

# draw the picture of the bulkloading key 
prob_dict = {}
for i, idx in enumerate(bin_idxs):
    prob_dict[idx] = ini_prob[i]
y = [ prob_dict[i] if i in prob_dict else 0.0 for i in range(min_bin_num, max_bin_num+1) ]

# %%
# # draw the estimated distribution and store the figure

# labels = x
# max_value = max(y)
# rounded_max_value = np.ceil(max_value * 50) / 50
# # rounded 0.05
# y_ = [ i * 100 for i in y]
# ax.set_xlabel('#Bin', fontsize = 15, fontname='Times New Roman')
# ax.set_ylabel('Prob (%)', fontsize = 15, fontname='Times New Roman')
# fig, ax = plt.subplots(figsize=(4, 2.5))
# ax.bar(labels, y_, color='#1F77B4', alpha=0.7)
# ax.set_xlabel('#Bin', fontsize = 15)
# ax.set_ylabel('Prob (%)', fontsize = 15)
# ax.set_ylim(0, 2)  # Set y-axis limits from 0 to 1
# ax.set_xlim(-10, 310)
# ax.grid(axis='y')
# fig.tight_layout()
# filepath = "/users/lingze/neurbench/data/keys/osm_cellids_200M_uint64_distribution_drift_07.pdf"
# fig.savefig(filepath, dpi = 800, bbox_inches = 'tight', pad_inches=0)
# # Show the plot
# fig.show()

# %%
filepath = "/users/lingze/neurbench/data/draw/osm_cellid_drift_07.npz"
x = np.array(x)
y = np.array(y)
np.savez(filepath, x=x, y=y)


