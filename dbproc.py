import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

sys.path.append("drift_ddpm")

import deterministic
import numpy as np
from tqdm import tqdm

deterministic.seed_everything(42)

import argparse
import os
import warnings

import data_utils as du
import lib_oversampling as lo
import pandas as pd
import torch

warnings.filterwarnings("ignore")


INFERENCE_BATCH_SIZE = 524288  # 262144


class ControllerDimAdapter(torch.nn.Module):
    """Wrapper to adapt controller from one dimension to another.

    This allows using a controller trained on one table (e.g., aka_title with d_in=2)
    on another table with different dimensions (e.g., aka_name with d_in=1).
    """
    def __init__(self, controller, source_dim: int, target_dim: int):
        super().__init__()
        self.controller = controller
        self.source_dim = source_dim
        self.target_dim = target_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor, drift: float):
        batch_size = x.shape[0]

        if self.target_dim < self.source_dim:
            # Target has fewer dims: pad input with zeros, then truncate output
            padding = torch.zeros(batch_size, self.source_dim - self.target_dim, device=x.device)
            x_padded = torch.cat([x, padding], dim=1)
            out = self.controller(x_padded, t, drift)
            return out[:, :self.target_dim]
        elif self.target_dim > self.source_dim:
            # Target has more dims: truncate input, then pad output with zeros
            x_truncated = x[:, :self.source_dim]
            out = self.controller(x_truncated, t, drift)
            padding = torch.zeros(batch_size, self.target_dim - self.source_dim, device=out.device)
            return torch.cat([out, padding], dim=1)
        else:
            # Same dimension
            return self.controller(x, t, drift)

    def to(self, device):
        self.controller = self.controller.to(device)
        return super().to(device)


def main(args: argparse.Namespace):
    print("=" * 80)
    print(f"[TIMING] Starting data generation pipeline at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    total_start = time.time()

    save_dir = os.path.join("expdir", args.dataset_name, args.table_name)
    if args.variant_id > 0:
        save_dir += f"-{args.variant_id}"

    print("save_dir", save_dir)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.device}")

    base_dir = os.path.join("datasets", args.dataset_name)

    config: dict = du.load_json(os.path.join(base_dir, "dataset_info.json"))
    config = config[args.table_name]

    train_data_path = os.path.join(base_dir, f"{args.table_name}.csv")
    # Try different CSV parsing strategies (C engine only for speed)
    original_data = None
    for strategy in [
        {"doublequote": True},                        # Standard CSV (most common)
        {"doublequote": False, "escapechar": "\\"},   # Backslash escaped
    ]:
        try:
            original_data = pd.read_csv(
                train_data_path, low_memory=False, on_bad_lines='warn', **strategy
            )
            print(f"Loaded with strategy: {strategy}")
            break
        except Exception as e:
            print(f"Strategy {strategy} failed: {e}")
            continue
    if original_data is None:
        raise RuntimeError(f"Failed to load {train_data_path} - check CSV format")
    print("Original data")
    print(original_data)

    train_data = original_data[config["applicable_columns"]]
    print("Data with drifting columns")
    print(train_data)

    # Reference dataset: use specified reference or default to {dataset_name}
    if args.reference_dataset:
        real_base_dir = os.path.join("datasets", args.reference_dataset)
    else:
        real_base_dir = os.path.join("datasets", args.dataset_name)

    print(f"Using reference dataset: {real_base_dir}")

    real_drift_data_path = os.path.join(real_base_dir, f"{args.table_name}.csv")
    original_real_drift_data = None
    for strategy in [
        {"doublequote": True},
        {"doublequote": False, "escapechar": "\\"},
    ]:
        try:
            original_real_drift_data = pd.read_csv(
                real_drift_data_path, low_memory=False, on_bad_lines='warn', **strategy
            )
            break
        except:
            continue
    if original_real_drift_data is None:
        raise RuntimeError(f"Failed to load {real_drift_data_path}")
    print("Real Drift data")
    print(original_real_drift_data)

    real_data = original_real_drift_data[config["applicable_columns"]]
    print("Real Drift Data with drifting columns")
    print(real_data)

    real_cond_data_path = os.path.join(
        real_base_dir, f"{args.table_name}.csv"
    )  # {args.table_name}
    original_real_cond_data = None
    for strategy in [
        {"doublequote": True},
        {"doublequote": False, "escapechar": "\\"},
    ]:
        try:
            original_real_cond_data = pd.read_csv(
                real_cond_data_path, low_memory=False, on_bad_lines='warn', **strategy
            )
            break
        except:
            continue
    if original_real_cond_data is None:
        raise RuntimeError(f"Failed to load {real_cond_data_path}")
    print("Conditional data")
    print(original_real_cond_data)

    # condition_data = original_real_cond_data[["id"]] ## TODO: for different table need to load different column
    # synthetic_data = original_data[["movie_id"]]

    if args.fillna:
        train_data.fillna(0.0, inplace=True)
        real_data.fillna(0.0, inplace=True)

    if args.reuse and os.path.exists(os.path.join(save_dir, "data_wrapper.pkl")):
        with open(os.path.join(save_dir, "data_wrapper.pkl"), "rb") as f:
            data_wrapper = pickle.load(f)
    else:
        data_wrapper = du.DataWrapper()
        data_wrapper.fit(train_data)

        with open(os.path.join(save_dir, "data_wrapper.pkl"), "wb") as f:
            pickle.dump(data_wrapper, f)

    if args.reuse and os.path.exists(os.path.join(save_dir, "train_x.npy")):
        with open(os.path.join(save_dir, "train_x.npy"), "rb") as f:
            train_x = np.load(f)
    else:
        train_x = data_wrapper.transform(train_data)

        with open(os.path.join(save_dir, "train_x.npy"), "wb") as f:
            np.save(f, train_x)

    if args.reuse and os.path.exists(os.path.join(save_dir, "real_data_wrapper.pkl")):
        with open(os.path.join(save_dir, "real_data_wrapper.pkl"), "rb") as f:
            real_data_wrapper = pickle.load(f)
    else:
        real_data_wrapper = du.DataWrapper()
        real_data_wrapper.fit(real_data)

        with open(os.path.join(save_dir, "real_data_wrapper.pkl"), "wb") as f:
            pickle.dump(real_data_wrapper, f)

    if args.reuse and os.path.exists(os.path.join(save_dir, "real_x.npy")):
        with open(os.path.join(save_dir, "real_x.npy"), "rb") as f:
            real_x = np.load(f)
    else:
        real_x = real_data_wrapper.transform(real_data)

        with open(os.path.join(save_dir, "real_x.npy"), "wb") as f:
            np.save(f, real_x)

    # if args.reuse and os.path.exists(os.path.join(save_dir, "cond_data_wrapper.pkl")):
    #     with open(os.path.join(save_dir, "cond_data_wrapper.pkl"), "rb") as f:
    #         cond_data_wrapper = pickle.load(f)
    # else:
    #     cond_data_wrapper = du.DataWrapper()
    #     cond_data_wrapper.fit(condition_data)

    #     with open(os.path.join(save_dir, "cond_data_wrapper.pkl"), "wb") as f:
    #         pickle.dump(cond_data_wrapper, f)

    # if args.reuse and os.path.exists(os.path.join(save_dir, "cond_x.npy")):
    #     with open(os.path.join(save_dir, "cond_x.npy"), "rb") as f:
    #         cond_x = np.load(f)
    # else:
    #     cond_x = cond_data_wrapper.transform(condition_data)

    #     with open(os.path.join(save_dir, "cond_x.npy"), "wb") as f:
    #         np.save(f, cond_x)

    # synthetic_data_wrapper = du.DataWrapper()
    # synthetic_data_wrapper.fit(synthetic_data)
    # synthetic_x = synthetic_data_wrapper.transform(synthetic_data)

    """ diffuser training. To avoid randomness, reseed everything. """
    deterministic.seed_everything(args.random_state)

    if not args.retrain_diffuser and os.path.exists(
        os.path.join(save_dir, "diffuser.pt")
    ):
        print("Load existing diffuser")
    else:
        print("Train diffuser")
        diffuser_start = time.time()
        lo.diffuser_training(
            train_x=train_x,
            save_path=os.path.join(save_dir, "diffuser.pt"),
            device=device,
            d_hidden=args.diffuser_dim,
            num_timesteps=args.diffuser_timesteps,
            epochs=args.diffuser_steps,
            lr=args.diffuser_lr,
            drop_out=0.0,
            bs=args.diffuser_bs,
            lambda_p=args.lambda_p,
            lambda_s=args.lambda_s,
        )
        diffuser_end = time.time()
        print(f"[TIMING] Diffuser training took: {diffuser_end - diffuser_start:.2f} seconds")

    diffuser = torch.load(os.path.join(save_dir, "diffuser.pt"))

    """ controller training. To avoid randomness, reseed everything. """
    deterministic.seed_everything(args.random_state)

    # Determine controller source
    if args.controller_from:
        # Use controller from another table
        controller_source_dir = os.path.join("expdir", args.dataset_name, args.controller_from)
        controller_path = os.path.join(controller_source_dir, "controller.pt")
        if not os.path.exists(controller_path):
            raise FileNotFoundError(
                f"Controller not found at {controller_path}. "
                f"Please train the controller for '{args.controller_from}' first."
            )
        print(f"Using controller from table '{args.controller_from}': {controller_path}")

        # Get source table's dimension from dataset_info.json
        full_config = du.load_json(os.path.join(base_dir, "dataset_info.json"))
        source_config = full_config.get(args.controller_from)
        if source_config is None:
            raise ValueError(f"Table '{args.controller_from}' not found in dataset_info.json")
        source_dim = len(source_config["applicable_columns"])
        target_dim = len(config["applicable_columns"])

        print(f"Source dim ({args.controller_from}): {source_dim}, Target dim ({args.table_name}): {target_dim}")

        raw_controller = torch.load(controller_path)
        if source_dim != target_dim:
            print(f"Wrapping controller with dimension adapter: {source_dim} -> {target_dim}")
            controller = ControllerDimAdapter(raw_controller, source_dim, target_dim)
        else:
            controller = raw_controller
    else:
        # Train or load controller for current table
        if not args.retrain_controller and os.path.exists(
            os.path.join(save_dir, "controller.pt")
        ):
            print("Load existing controller")
        else:
            print("Train controller")
            controller_start = time.time()
            lo.controller_training(
                train_x=train_x,
                real_x=real_x,
                # cond_x=cond_x,
                # synthetic_x=synthetic_x,
                diffuser=diffuser,
                save_path=os.path.join(save_dir, "controller.pt"),
                cond_save_path=os.path.join(save_dir, "controller_cond.pt"),
                device=device,
                lr=args.controller_lr,
                d_hidden=args.controller_dim,
                steps=args.controller_steps,
                drop_out=0.0,
                bs=args.controller_bs,
                # New parameters for better training
                drift_range=(args.drift_range_min, args.drift_range_max),
                loss_weight_corr=args.loss_weight_corr,
                loss_weight_real=args.loss_weight_real,
            )
            controller_end = time.time()
            print(f"[TIMING] Controller training took: {controller_end - controller_start:.2f} seconds")

        controller = torch.load(os.path.join(save_dir, "controller.pt"))
    # controller_cond = torch.load(os.path.join(save_dir, "controller_cond.pt"))

    """ oversampling. To avoid randomness, reseed everything. """
    deterministic.seed_everything(args.random_state)

    ids = range(config["n_samples"])
    batched_ids = [
        ids[x : x + INFERENCE_BATCH_SIZE]
        for x in range(0, len(ids), INFERENCE_BATCH_SIZE)
    ]

    print(f"[TIMING] Starting data generation for {config['n_samples']} samples in {len(batched_ids)} batches...")
    oversampling_start = time.time()

    num_gpus = args.num_gpus
    if num_gpus > 1 and len(batched_ids) > 1:
        # Multi-GPU parallel generation
        gpu_ids = [args.device + i for i in range(num_gpus)]
        print(f"[Parallel] Using {num_gpus} GPUs: {gpu_ids}")

        # Prepare model copies for each GPU
        diffuser_copies = {}
        controller_copies = {}
        for gpu_id in gpu_ids:
            gpu_device = torch.device(f"cuda:{gpu_id}")
            diffuser_copies[gpu_id] = copy.deepcopy(diffuser).to(gpu_device)
            diffuser_copies[gpu_id].variables_to_device(gpu_device)
            controller_copies[gpu_id] = copy.deepcopy(controller).to(gpu_device)

        def generate_batch(batch_idx, batch, gpu_id):
            """Worker function to generate a batch on a specific GPU."""
            gpu_device = torch.device(f"cuda:{gpu_id}")
            batch_start = time.time()
            sample_data = lo.oversampling(
                len(batch),
                controller_copies[gpu_id],
                diffuser_copies[gpu_id],
                None,
                None,
                None,
                gpu_device,
                args.drift,
                args.scale_factor,
            )
            # Move result to CPU immediately to free GPU memory
            result = sample_data.cpu()
            batch_end = time.time()
            print(f"[TIMING] Batch {batch_idx + 1}/{len(batched_ids)} (GPU {gpu_id}) took: {batch_end - batch_start:.2f}s")
            return batch_idx, result

        # Submit all batches to thread pool
        all_data = [None] * len(batched_ids)
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for batch_idx, b in enumerate(batched_ids):
                gpu_id = gpu_ids[batch_idx % num_gpus]
                futures.append(executor.submit(generate_batch, batch_idx, b, gpu_id))

            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                batch_idx, result = future.result()
                all_data[batch_idx] = result

        sample_data = torch.cat(all_data, dim=0)

        # Cleanup GPU copies
        del diffuser_copies, controller_copies
        torch.cuda.empty_cache()
    else:
        # Single GPU sequential generation
        all_data = []
        for batch_idx, b in enumerate(tqdm(batched_ids)):
            batch_start = time.time()
            sample_data = lo.oversampling(
                len(b),
                controller,
                diffuser,
                None,  # controller_cond,
                None,  # cond_x,
                None,  # synthetic_x,
                device,
                args.drift,
                args.scale_factor,
            )
            batch_end = time.time()
            print(f"[TIMING] Batch {batch_idx + 1}/{len(batched_ids)} generation took: {batch_end - batch_start:.2f} seconds")
            all_data.append(sample_data)

        sample_data = torch.cat(all_data, dim=0)

    oversampling_end = time.time()
    print(f"[TIMING] Total oversampling took: {oversampling_end - oversampling_start:.2f} seconds")

    sample_data = sample_data.cpu().numpy()

    # Set reference distribution for frequency-preserving sampling
    # This ensures generated ID columns match the target data's frequency distribution
    print("\n[Frequency Preservation] Setting reference distributions...")

    # Mapping from foreign key column to primary table
    FK_TO_TABLE = {
        'movie_id': 'title',
        'person_id': 'name',
        'company_id': 'company_name',
        'keyword_id': 'keyword',
        'linked_movie_id': 'title',
        'link_type_id': 'link_type',
        'info_type_id': 'info_type',
        'kind_id': 'kind_type',
        'role_id': 'role_type',
    }

    # Check if we have a real reference dataset (different from original)
    has_real_reference = (args.reference_dataset and
                          args.reference_dataset != args.dataset_name)

    for col in config["applicable_columns"]:
        if col in data_wrapper.num_normalizer:
            # Only use frequency preservation for foreign key columns
            if col in FK_TO_TABLE:
                if has_real_reference:
                    # Use real reference data for frequency shape
                    freq_shape_data = real_data[col].dropna().values
                    # Use the unique IDs from ref data directly as valid_ids
                    # This preserves:
                    # 1. Frequency distribution shape (hotspot effect)
                    # 2. Cross-table correlation (e.g., movie_keyword and movie_companies
                    #    share ~82% of movie_ids in real data, random sampling would break this)
                    # 3. Valid FK references
                    valid_ids = np.unique(freq_shape_data)
                    print(f"  {col}: FK column, using {len(valid_ids)} unique IDs from --ref "
                          f"(total {len(freq_shape_data)} values)")
                    data_wrapper.set_reference_distribution(col, freq_shape_data, valid_ids)
                else:
                    # No real reference: use temperature scaling to synthesize target distribution
                    original_col_data = train_data[col].dropna().values
                    print(f"  {col}: FK column, synthesizing target distribution with drift={args.drift}")
                    data_wrapper.set_synthetic_reference_distribution(
                        col, original_col_data, args.drift, mode='auto'
                    )
            else:
                # Non-FK columns (like production_year): skip frequency preservation
                # Let the model's inverse_transform handle the drift naturally
                print(f"  {col}: not a FK column, skip frequency preservation")

    sample_data = data_wrapper.Reverse(sample_data)
    sample_data = sample_data[config["applicable_columns"]]

    # if len(original_data.index) > len(sample_data.index):
    # original_data = original_data[:len(sample_data.index)]

    if len(original_real_drift_data.index) > len(sample_data.index):
        original_real_drift_data = original_real_drift_data[: len(sample_data.index)]

    print("Drifted columns")
    print(sample_data)

    # original_data[config["applicable_columns"]] = sample_data
    original_real_drift_data[config["applicable_columns"]] = sample_data
    print("Drifted data")
    print(original_real_drift_data)

    original_real_drift_data.to_csv(
        os.path.join(save_dir, f"{args.table_name}.drifted.csv"),
        index=False,
        doublequote=False,
        escapechar="\\",
    )

    total_end = time.time()
    total_time = total_end - total_start
    print("=" * 80)
    print(f"[TIMING] Data generation pipeline completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[TIMING] Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="imdb")
    parser.add_argument("--table-name", type=str, default="nosuchtable")
    parser.add_argument(
        "--reference-dataset",
        type=str,
        default=None,
        help="Reference dataset for drift direction (default: {dataset_name})"
    )

    parser.add_argument(
        "--diffuser-dim", nargs="+", type=int, default=(512, 1024, 1024, 512)
    )
    parser.add_argument("--diffuser-lr", type=float, default=0.0018)
    parser.add_argument("--diffuser-steps", type=int, default=30000)
    parser.add_argument("--diffuser-bs", type=int, default=2048)
    parser.add_argument("--diffuser-timesteps", type=int, default=1000)

    parser.add_argument(
        "--controller-dim", nargs="+", type=int, default=(512, 512)
    )  ## TODO: higher --controller-dim 1024 1024  512 512 512
    parser.add_argument("--controller-lr", type=float, default=0.001)  ## TODO: lower
    parser.add_argument("--controller-steps", type=int, default=10000)  ## TODO: higher
    parser.add_argument("--controller-bs", type=int, default=512)

    parser.add_argument("--device", type=int, default=1)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel batch generation (default: 1)"
    )
    parser.add_argument("--scale-factor", type=float, default=8.0)
    # parser.add_argument("--save-name", type=str, default="output")

    parser.add_argument("--lambda-p", type=float, default=1.0)
    parser.add_argument("--lambda-s", type=float, default=1.0)

    parser.add_argument("--retrain-diffuser", action="store_true")
    parser.add_argument("--retrain-controller", action="store_true")

    parser.add_argument("--reuse", action="store_true")

    parser.add_argument("--variant-id", type=int, default=-1)

    parser.add_argument("--drift", type=float, default=0.3)

    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--fillna", action="store_true", default=False)

    parser.add_argument(
        "--controller-from",
        type=str,
        default=None,
        help="Use controller from another table (e.g., 'aka_title'), diffuser still uses current table"
    )

    # Controller training improvement parameters
    parser.add_argument("--drift-range-min", type=float, default=0.05,
                        help="Min drift for controller training (default: 0.05)")
    parser.add_argument("--drift-range-max", type=float, default=0.75,
                        help="Max drift for controller training (default: 0.75)")
    parser.add_argument("--loss-weight-corr", type=float, default=0.8,
                        help="Weight for correlation loss (default: 0.8)")
    parser.add_argument("--loss-weight-real", type=float, default=0.1,
                        help="Weight for RealMSE loss (default: 0.1)")

    # parser.add_argument("--cond", action="store_true", default=False)

    args = parser.parse_args()

    main(args)
