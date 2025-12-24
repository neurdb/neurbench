import json
import pickle

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from torch.utils.data import Dataset


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def save_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def solve_target_distribution(original_data: np.ndarray, target_drift: float,
                               mode: str = 'sharpen') -> np.ndarray:
    """
    Generate a synthetic reference distribution with a specific JS divergence from original.

    Uses temperature scaling: Q_i ∝ P_i^(1/τ)
    - τ = 1.0: distribution unchanged (JS ≈ 0)
    - τ < 1.0 (sharpen/cool): head effect stronger, rich get richer
    - τ > 1.0 (flatten/heat): distribution flatter, long tail effect

    Args:
        original_data: 1D array of values from original dataset (e.g., movie_ids)
        target_drift: Target JS divergence (e.g., 0.3)
        mode: 'sharpen' (head more concentrated), 'flatten' (more uniform),
              or 'auto' (try both, pick closer)

    Returns:
        synthetic_ref_data: Array that can be passed to set_reference_distribution
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.special import softmax

    # 1. Get original probability distribution P
    unique_vals, counts = np.unique(original_data, return_counts=True)
    p_probs = counts / counts.sum()

    # 2. Define function: input temperature T, return JS divergence
    def get_js_dist(t):
        if t <= 0:
            return 1.0
        # Temperature scaling: q_i = p_i^(1/t) / Z
        # Compute in log domain for numerical stability
        log_p = np.log(p_probs + 1e-10)
        q_probs = softmax(log_p / t)
        return jensenshannon(p_probs, q_probs)

    # 3. Binary search to find optimal temperature T
    def find_temperature(low, high, target_js):
        best_t = 1.0
        for _ in range(30):  # 30 iterations for precision
            mid = (low + high) / 2
            current_js = get_js_dist(mid)

            if low < 1.0:  # sharpen mode: T < 1 means more peaked
                if current_js < target_js:
                    high = mid  # Need more peaked, smaller T
                else:
                    low = mid
            else:  # flatten mode: T > 1 means flatter
                if current_js < target_js:
                    low = mid  # Need flatter, larger T
                else:
                    high = mid
            best_t = mid
        return best_t, get_js_dist(best_t)

    # Try both modes if auto
    if mode == 'auto':
        t_sharpen, js_sharpen = find_temperature(0.01, 1.0, target_drift)
        t_flatten, js_flatten = find_temperature(1.0, 20.0, target_drift)

        # Pick the one closer to target
        if abs(js_sharpen - target_drift) < abs(js_flatten - target_drift):
            best_t, best_js = t_sharpen, js_sharpen
            mode_used = 'sharpen'
        else:
            best_t, best_js = t_flatten, js_flatten
            mode_used = 'flatten'
    elif mode == 'sharpen':
        best_t, best_js = find_temperature(0.01, 1.0, target_drift)
        mode_used = 'sharpen'
    else:  # flatten
        best_t, best_js = find_temperature(1.0, 20.0, target_drift)
        mode_used = 'flatten'

    print(f"[solve_target_distribution] mode={mode_used}, T={best_t:.4f}, "
          f"target_JS={target_drift:.4f}, achieved_JS={best_js:.4f}")

    # 4. Generate target frequencies based on optimal T
    log_p = np.log(p_probs + 1e-10)
    target_probs = softmax(log_p / best_t)

    # Convert probabilities back to counts (preserve total rows)
    total_rows = len(original_data)
    target_counts = (target_probs * total_rows).astype(int)

    # Fix rounding errors
    diff = total_rows - target_counts.sum()
    if diff > 0:
        # Add to highest frequency items
        idx = np.argsort(-target_counts)[:diff]
        target_counts[idx] += 1
    elif diff < 0:
        # Remove from highest frequency items
        idx = np.argsort(-target_counts)[:-diff]
        target_counts[idx] -= 1

    # 5. Generate synthetic reference data (repeat each unique value by its target count)
    synthetic_ref_data = np.repeat(unique_vals, target_counts)

    return synthetic_ref_data


def merge_wrapper(wrappers):
    if len(wrappers) == 1:
        return wrappers[0]

    wrapper = DataWrapper()

    wrapper.raw_dim = 0
    wrapper.raw_columns = []
    wrapper.all_distinct_values = {}
    wrapper.num_normalizer = {}
    wrapper.num_dim = 0
    wrapper.columns = []
    wrapper.col_dim = []
    wrapper.col_dtype = {}

    num_cols = []
    cat_cols = []
    num_col_dims = []
    cat_col_dims = []
    for wp in wrappers:
        wrapper.raw_dim += wp.raw_dim
        wrapper.raw_columns += list(wp.raw_columns)
        wrapper.all_distinct_values.update(wp.all_distinct_values)
        wrapper.num_normalizer.update(wp.num_normalizer)
        wrapper.num_dim += wp.num_dim
        wrapper.col_dtype.update(wp.col_dtype)

        for i, col in enumerate(wp.columns):
            if col in wp.num_normalizer.keys():
                num_cols.append(col)
                num_col_dims.append(wp.col_dim[i])
            else:
                cat_cols.append(col)
                cat_col_dims.append(wp.col_dim[i])

    wrapper.columns = num_cols + cat_cols
    wrapper.col_dim = num_col_dims + cat_col_dims

    return wrapper


class DataWrapper:
    def __init__(self, num_encoder="quantile", seed=0, dequantize=True):
        self.num_encoder = num_encoder
        self.seed = seed
        self.dequantize = dequantize  # Enable dequantization for discrete numeric columns
        self._rng = np.random.RandomState(seed)
        # Reference distribution for frequency-preserving sampling
        self.reference_distribution = {}  # col -> (unique_values, cumulative_probs)

    def _dequantize(self, values):
        """Add uniform noise [0, 1) to discrete values to make them continuous.

        This is a standard technique for feeding discrete data into continuous models.
        E.g., movie_id=5 becomes 5.xx where xx is random, spreading it over [5, 6).
        """
        noise = self._rng.uniform(0, 1, values.shape)
        return values + noise

    def set_reference_distribution(self, col: str, reference_data: np.ndarray,
                                     valid_ids: np.ndarray = None):
        """Set reference distribution for frequency-preserving sampling.

        This enables mapping generated values to discrete IDs while preserving
        the frequency distribution of the reference (target) dataset.

        Args:
            col: Column name
            reference_data: 1D array of values from the reference dataset (for frequency shape)
            valid_ids: Optional 1D array of valid IDs to use instead of reference_data's IDs.
                      This allows using frequency shape from one dataset (e.g., 2016)
                      but ID values from another (e.g., 2017 title table).
        """
        # Backward compatibility: initialize if not exists (for old pickled objects)
        if not hasattr(self, 'reference_distribution'):
            self.reference_distribution = {}

        # Count frequency of each unique value in reference data
        ref_unique_vals, counts = np.unique(reference_data, return_counts=True)

        # Sort by frequency (descending) to get the frequency "shape"
        freq_sort_idx = np.argsort(-counts)  # Descending by frequency
        sorted_counts = counts[freq_sort_idx]

        if valid_ids is not None:
            valid_ids_unique = np.unique(valid_ids)
            n_valid = len(valid_ids_unique)
            n_ref = len(ref_unique_vals)

            # Check if valid_ids is the same set as ref unique values
            if n_valid == n_ref and set(valid_ids_unique) == set(ref_unique_vals):
                # Same IDs: preserve frequency-ID mapping from reference
                # Use ref_unique_vals sorted by frequency (descending)
                unique_vals = ref_unique_vals[freq_sort_idx]
                print(f"[DataWrapper] Set reference distribution for '{col}': "
                      f"{n_ref} unique IDs from ref, preserving freq-ID mapping, "
                      f"top freq={sorted_counts[0]:.0f}")
            else:
                # Different IDs: map frequency shape to new valid_ids
                np.random.shuffle(valid_ids_unique)  # Randomize which IDs get high frequency

                if n_valid != n_ref:
                    # Interpolate frequency shape to match valid_ids count
                    old_indices = np.linspace(0, n_ref - 1, n_ref)
                    new_indices = np.linspace(0, n_ref - 1, n_valid)
                    sorted_counts = np.interp(new_indices, old_indices, sorted_counts)
                    sorted_counts = np.maximum(sorted_counts, 1)  # At least 1 occurrence

                unique_vals = valid_ids_unique
                print(f"[DataWrapper] Set reference distribution for '{col}': "
                      f"shape from {n_ref} ref values, mapped to {n_valid} valid IDs")
        else:
            # Use both frequency and IDs from reference data
            unique_vals = ref_unique_vals[freq_sort_idx]
            print(f"[DataWrapper] Set reference distribution for '{col}': "
                  f"{len(unique_vals)} unique values, "
                  f"top freq={counts.max()}, min freq={counts.min()}")

        # Compute cumulative probabilities (CDF) from the frequency shape
        probs = sorted_counts / sorted_counts.sum()
        cumulative_probs = np.cumsum(probs)
        self.reference_distribution[col] = (unique_vals, cumulative_probs)

    def set_synthetic_reference_distribution(self, col: str, original_data: np.ndarray,
                                              target_drift: float, mode: str = 'sharpen'):
        """Set synthetic reference distribution using temperature scaling.

        When no real reference data is available, this method creates a synthetic
        target distribution with a specific JS divergence from the original.

        Args:
            col: Column name
            original_data: 1D array of values from original dataset
            target_drift: Target JS divergence (e.g., 0.3)
            mode: 'sharpen', 'flatten', or 'auto'
        """
        synthetic_ref = solve_target_distribution(original_data, target_drift, mode)
        self.set_reference_distribution(col, synthetic_ref)
        print(f"[DataWrapper] Synthetic reference for '{col}': "
              f"target_drift={target_drift}, mode={mode}")

    def fit(self, dataframe, all_category=False):
        self.raw_dim = dataframe.shape[1]
        self.raw_columns = dataframe.columns
        self.all_distinct_values = {}  # For categorical columns
        self.num_normalizer = {}  # For numerical columns
        self.num_dim = 0
        self.columns = []
        self.col_dim = []
        self.col_dtype = {}

        for i, col in enumerate(self.raw_columns):
            if all_category:
                break
            if is_numeric_dtype(dataframe[col]):
                col_data = dataframe.loc[pd.notna(dataframe[col])][col]
                self.col_dtype[col] = col_data.dtype
                if self.num_encoder == "quantile":
                    self.num_normalizer[col] = QuantileTransformer(
                        output_distribution="normal",
                        n_quantiles=max(min(len(col_data) // 30, 1000), 10),
                        subsample=1000000000,
                        random_state=self.seed,
                    )
                elif self.num_encoder == "standard":
                    self.num_normalizer[col] = StandardScaler()
                elif self.num_encoder == "minmax":
                    self.num_normalizer[col] = MinMaxScaler()
                else:
                    raise ValueError(f"Unknown num encoder: {self.num_encoder}")

                # Dequantize: add noise to discrete values before fitting
                fit_data = col_data.values.reshape(-1, 1)
                if self.dequantize and np.issubdtype(self.col_dtype[col], np.integer):
                    fit_data = self._dequantize(fit_data.astype(np.float64))

                self.num_normalizer[col].fit(fit_data)
                self.columns.append(col)
                self.num_dim += 1
                self.col_dim.append(1)

        for i, col in enumerate(self.raw_columns):
            if col not in self.num_normalizer.keys():
                col_data = dataframe.loc[pd.notna(dataframe[col])][col]
                self.col_dtype[col] = col_data.dtype
                distinct_values = col_data.unique()
                distinct_values.sort()
                self.all_distinct_values[col] = distinct_values
                self.columns.append(col)
                self.col_dim.append(max(1, int(np.ceil(np.log2(len(distinct_values))))))

    def transform(self, data):
        # normalize the numreical column and transform the categorical data to oridinal type
        reorder_data = data[self.columns].values
        norm_data = []

        for i, col in enumerate(self.columns):
            col_data = reorder_data[:, i]

            if col in self.all_distinct_values.keys():
                col_data = self.CatValsToNum(col, col_data).reshape(-1, 1)
                col_data = self.ValsToBit(col_data, self.col_dim[i])
                norm_data.append(col_data)

            elif col in self.num_normalizer.keys():
                # Dequantize: add noise to discrete values before transform
                transform_data = col_data.reshape(-1, 1)
                if self.dequantize and np.issubdtype(self.col_dtype[col], np.integer):
                    transform_data = self._dequantize(transform_data.astype(np.float64))
                norm_data.append(
                    self.num_normalizer[col]
                    .transform(transform_data)
                    .reshape(-1, 1)
                )

        norm_data = np.concatenate(norm_data, axis=1)
        norm_data = norm_data.astype(np.float32)

        return norm_data

    def ReOrderColumns(self, data: pd.DataFrame):
        ndf = pd.DataFrame([])
        for col in self.raw_columns:
            ndf[col] = data[col]
        return ndf

    def GetColData(self, data, col_id):
        col_index = np.cumsum(self.col_dim)
        col_data = data.copy()
        if col_id == 0:
            return col_data[:, : col_index[0]]
        else:
            return col_data[:, col_index[col_id - 1] : col_index[col_id]]

    def ValsToBit(self, values, bits):
        bit_values = np.zeros((values.shape[0], bits))
        for i in range(values.shape[0]):
            bit_val = np.mod(
                np.right_shift(int(values[i]), list(reversed(np.arange(bits)))), 2
            )
            bit_values[i, :] = bit_val
        return bit_values

    def BitsToVals(self, bit_values):
        bits = bit_values.shape[1]
        values = bit_values.astype(int)
        values = values * (2 ** np.array(list((reversed(np.arange(bits))))))
        values = np.sum(values, axis=1)
        return values

    def CatValsToNum(self, col, values):
        num_values = pd.Categorical(
            values, categories=self.all_distinct_values[col]
        ).codes
        # num_values = np.zeros_like(values)
        # for i, val in enumerate(values):
        # 	ind = np.where(self.all_distinct_values[col] == val)
        # 	num_values[i] = ind[0][0]
        return num_values

    def NumValsToCat(self, col, values):
        cat_values = np.zeros_like(values).astype(object)
        # print(col_name, values)
        values = np.clip(values, 0, len(self.all_distinct_values[col]) - 1)
        for i, val in enumerate(values):
            # val = np.clip(val, self.Mins[col_id], self.Maxs[col_id])
            cat_values[i] = self.all_distinct_values[col][int(val)]
        return cat_values

    def ReverseToOrdi(self, data):
        reverse_data = []

        # Unnorm the normalized numerical columns, and reverse the binary code to ordinal columns
        for i, col in enumerate(self.columns):
            # print(col_name)
            col_data = self.GetColData(data, i)
            if col in self.all_distinct_values.keys():
                col_data = np.round(col_data)
                col_data = self.BitsToVals(col_data)
                col_data = col_data.astype(np.int32)
            else:
                # Check if we have reference distribution for frequency-preserving sampling
                if col in self.reference_distribution:
                    col_data = self._reverse_with_reference_distribution(col, col_data)
                else:
                    col_data = self.num_normalizer[col].inverse_transform(
                        col_data.reshape(-1, 1)
                    )
                    if np.issubdtype(self.col_dtype[col], np.integer):
                        # Use floor instead of round for dequantized data
                        col_data = np.floor(col_data).astype(int)
                # col_data = self.NumValsToCat(col, col_data)
            # col_data = col_data.astype(self.raw_data[col].dtype)
            reverse_data.append(col_data.reshape(-1, 1))
        reverse_data = np.concatenate(reverse_data, axis=1)
        return reverse_data

    def _reverse_with_reference_distribution(self, col: str, normalized_data: np.ndarray):
        """Reverse normalized data using EMPIRICAL RANKING to guarantee exact distribution match.

        Uses empirical ranking instead of theoretical normal CDF to handle variance shrinkage
        in diffusion outputs. This ensures:
        - Diffusion output ordering is preserved (ranks come from diffusion)
        - Exact frequency distribution match with reference data
        """
        unique_vals, cumulative_probs = self.reference_distribution[col]

        data_flat = normalized_data.flatten()
        n_samples = len(data_flat)

        # Use empirical ranking: argsort twice gives rank of each element
        # e.g., [-2.5, 0.1, 1.5] -> ranks [0, 1, 2] -> quantiles [0.0, 0.5, 1.0]
        # This uniformly covers [0, 1] regardless of variance shrinkage
        ranks = np.argsort(np.argsort(data_flat))
        quantiles = ranks / (n_samples - 1) if n_samples > 1 else np.zeros(n_samples)

        # Map quantiles to discrete values using reference CDF
        indices = np.searchsorted(cumulative_probs, quantiles)
        indices = np.clip(indices, 0, len(unique_vals) - 1)

        result = unique_vals[indices]

        return result.reshape(-1, 1)

    def ReverseToCat(self, data):
        reverse_data = []
        for i, col in enumerate(self.columns):
            col_data = data[:, i]
            if col in self.all_distinct_values.keys():
                col_data = self.NumValsToCat(col, col_data)
            reverse_data.append(col_data.reshape(-1, 1))
        reverse_data = np.concatenate(reverse_data, axis=1)
        return reverse_data

    def Reverse(self, data):
        data = self.ReverseToOrdi(data)
        data = self.ReverseToCat(data)
        data = pd.DataFrame(data, columns=self.columns)
        return self.ReOrderColumns(data)

    def RejectSample(self, sample):
        all_index = set(range(sample.shape[0]))
        allow_index = set(range(sample.shape[0]))
        for i, col in enumerate(self.columns):
            if col in self.all_distinct_values.keys():
                allow_index = allow_index & set(
                    np.where(sample[:, i] < len(self.all_distinct_values[col]))[0]
                )
                allow_index = allow_index & set(np.where(sample[:, i] >= 0)[0])
        reject_index = all_index - allow_index
        allow_index = np.array(list(allow_index))
        reject_index = np.array(list(reject_index))
        # allow_sample = sample[allow_index, :]
        return allow_index, reject_index


class TableDataset(Dataset):
    def __init__(self, data):
        super().__init__()

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return self.data[ind, :]


def cycle(dl):
    while True:
        for data in dl:
            yield data


def prepare_fast_dataloader(D, shuffle: bool, batch_size: int):
    dataloader = FastTensorDataLoader(D, batch_size=batch_size, shuffle=shuffle)
    while True:
        yield from dataloader


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        # assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.n_dim = tensors[0].shape[0]
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def miss_col(dataframe):
    for col in dataframe.columns:
        print(col, pd.isna(dataframe[col]).sum() / len(dataframe))


def generate_pk_weights(
    table, pk_name, removal_attr, removal_attr_values, attr_type="Numerical"
):
    project_tb = table.loc[:, [pk_name, removal_attr]]
    weights = None
    if attr_type == "Numerical":
        pk_values = project_tb[pk_name].values
        weights = project_tb[removal_attr].values
    elif attr_type == "Categorical":
        select_count = (
            project_tb.loc[project_tb[removal_attr].isin(removal_attr_values)]
            .groupby(pk_name)
            .count()
            .reset_index()
        )
        pk_values = select_count[pk_name].values
        weights = select_count[removal_attr].values

        remaining_ids = np.array(list(set(project_tb[pk_name]).difference(pk_values)))
        pk_values = np.concatenate((pk_values, remaining_ids))
        weights = np.concatenate((weights, np.zeros(len(remaining_ids))))
    else:
        raise ValueError(f"Unknown attributes type {attr_type}")
    weights = weights.astype(np.float64)
    return pk_values, weights


def index_to_onehot(x, n_classes):
    x_onehot = np.zeros((x.shape[0], n_classes))
    idx = np.arange(len(x_onehot))
    x_onehot[idx, x.astype(int).reshape(1, -1)] = 1
    return x_onehot
