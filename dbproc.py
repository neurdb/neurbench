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
import drift_ops as lo
import pandas as pd
import torch

warnings.filterwarnings("ignore")


INFERENCE_BATCH_SIZE = 524288  # 262144


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Smart CSV loader that auto-detects the correct quoting strategy.
    If first strategy has warnings/skipped lines, try next one.
    """
    # Use escapechar first for: .drifted.csv files OR imdb dataset (without year)
    use_escapechar_first = path.endswith(".drifted.csv") or "/imdb/" in path or "\\imdb\\" in path

    if use_escapechar_first:
        strategies = [
            {"doublequote": False, "escapechar": "\\"},   # Backslash escaped
            {"doublequote": True},                        # Standard CSV (fallback)
        ]
    else:
        strategies = [
            {"doublequote": True},                        # Standard CSV (most common)
            {"doublequote": False, "escapechar": "\\"},   # Backslash escaped
        ]

    last_df = None
    for strategy in strategies:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                df = pd.read_csv(path, low_memory=False, on_bad_lines='warn', **strategy, **kwargs)

            # If no warnings, this strategy works - return immediately
            if len(w) == 0:
                return df
            # Otherwise save and try next strategy
            last_df = df
        except Exception:
            continue

    # All strategies had warnings, return last successful one
    if last_df is not None:
        return last_df
    raise RuntimeError(f"Failed to load {path}")


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

    # Timing tracking
    timing = {
        "diffuser_train": 0.0,
        "controller_train": 0.0,
        "generation": 0.0,
        "total": 0.0,
    }

    # Build save_dir with reference_dataset/variant_id suffix to avoid overwriting
    # e.g., expdir/imdb_ref_imdb_2017/aka_name/ or expdir/imdb-1/aka_name/
    dataset_dir = args.dataset_name
    if args.reference_dataset and args.reference_dataset != args.dataset_name:
        dataset_dir = f"{args.dataset_name}_ref_{args.reference_dataset}"
    if args.variant_id > 0:
        dataset_dir += f"-{args.variant_id}"

    save_dir = os.path.join("expdir", dataset_dir, args.table_name)

    print("save_dir", save_dir)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.device}")

    base_dir = os.path.join("datasets", args.dataset_name)

    config: dict = du.load_json(os.path.join(base_dir, "dataset_info.json"))
    config = config[args.table_name]

    train_data_path = os.path.join(base_dir, f"{args.table_name}.csv")
    original_data = load_csv(train_data_path)
    print("Original data")
    print(original_data)

    train_data = original_data[config["applicable_columns"]]
    print("Data with drifting columns")
    print(train_data)

    # Reference dataset: use specified reference or default to {dataset_name}
    # Check if we have a meaningful reference dataset (different from source)
    has_reference = args.reference_dataset and args.reference_dataset != args.dataset_name
    if args.reference_dataset:
        real_base_dir = os.path.join("datasets", args.reference_dataset)
    else:
        real_base_dir = os.path.join("datasets", args.dataset_name)

    print(f"Using reference dataset: {real_base_dir}")

    # Disable RealMSE loss if no meaningful reference dataset
    if not has_reference:
        if args.loss_weight_real > 0:
            print(f"[INFO] No reference dataset specified (--ref), disabling RealMSE loss (was {args.loss_weight_real})")
            args.loss_weight_real = 0.0

    real_drift_data_path = os.path.join(real_base_dir, f"{args.table_name}.csv")
    original_real_drift_data = load_csv(real_drift_data_path)
    print("Real Drift data")
    print(original_real_drift_data)

    real_data = original_real_drift_data[config["applicable_columns"]]
    print("Real Drift Data with drifting columns")
    print(real_data)

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
        timing["diffuser_train"] = diffuser_end - diffuser_start
        print(f"[TIMING] Diffuser training took: {timing['diffuser_train']:.2f} seconds")

    diffuser = torch.load(os.path.join(save_dir, "diffuser.pt"))

    # Override num_timesteps for faster sampling (model was trained with 1000, but can sample with fewer)
    if diffuser.num_timesteps != args.diffuser_timesteps:
        print(f"[Sampling] Overriding timesteps: {diffuser.num_timesteps} -> {args.diffuser_timesteps}")
        diffuser.num_timesteps = args.diffuser_timesteps

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
                loss_weight_drift=args.loss_weight_drift,
                loss_weight_corr=args.loss_weight_corr,
                loss_weight_real=args.loss_weight_real,
            )
            controller_end = time.time()
            timing["controller_train"] = controller_end - controller_start
            print(f"[TIMING] Controller training took: {timing['controller_train']:.2f} seconds")

        controller = torch.load(os.path.join(save_dir, "controller.pt"))
    # controller_cond = torch.load(os.path.join(save_dir, "controller_cond.pt"))

    # --train-only: exit after training, skip generation
    if args.train_only:
        total_end = time.time()
        timing["total"] = total_end - total_start

        print("=" * 80)
        print("[TIMING SUMMARY] (train-only mode)")
        print(f"  Diffuser Training:   {timing['diffuser_train']:.2f} seconds")
        print(f"  Controller Training: {timing['controller_train']:.2f} seconds")
        print(f"  Training Total:      {timing['diffuser_train'] + timing['controller_train']:.2f} seconds")
        print(f"  Total:               {timing['total']:.2f} seconds")
        print("=" * 80)
        print(f"[TIMING] Training completed (train-only mode) at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return

    """ oversampling. To avoid randomness, reseed everything. """
    deterministic.seed_everything(args.random_state)

    # Support chunked generation with --sample-start and --sample-count
    sample_start = args.sample_start
    is_chunked = args.sample_count > 0

    if is_chunked:
        # Chunked mode: use provided sample_count directly
        sample_count = args.sample_count
        sample_end = sample_start + sample_count
        actual_count = sample_count
    else:
        # Non-chunked: generate all samples based on reference data size
        total_samples = len(original_real_drift_data)
        sample_count = total_samples - sample_start
        sample_end = total_samples
        actual_count = sample_count

    if is_chunked or sample_start > 0:
        print(f"[Chunk] Generating samples {sample_start} to {sample_end} ({actual_count} samples)")

    ids = range(actual_count)
    batched_ids = [
        ids[x : x + INFERENCE_BATCH_SIZE]
        for x in range(0, len(ids), INFERENCE_BATCH_SIZE)
    ]

    print(f"[TIMING] Starting data generation for {actual_count} samples in {len(batched_ids)} batches...")
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
                sample_steps=args.sample_steps,
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
                sample_steps=args.sample_steps,
            )
            batch_end = time.time()
            print(f"[TIMING] Batch {batch_idx + 1}/{len(batched_ids)} generation took: {batch_end - batch_start:.2f} seconds")
            all_data.append(sample_data)

        sample_data = torch.cat(all_data, dim=0)

    oversampling_end = time.time()
    timing["generation"] = oversampling_end - oversampling_start
    print(f"[TIMING] Total oversampling took: {timing['generation']:.2f} seconds")

    sample_data = sample_data.cpu().numpy()

    # For chunked generation: save normalized data, skip Frequency Preservation
    # Frequency Preservation will be done after all chunks are merged in cli.py
    if is_chunked:
        npy_filename = f"{args.table_name}.normalized.chunk_{sample_start}_{sample_end}.npy"
        np.save(os.path.join(save_dir, npy_filename), sample_data)
        print(f"[Chunk] Saved normalized data to {npy_filename}")

        # Save CSV without frequency preservation (for debugging/fallback)
        simple_data = data_wrapper.Reverse(sample_data, skip_freq_preservation=True)
        simple_data = simple_data[config["applicable_columns"]]

        original_real_drift_data_chunk = original_real_drift_data.iloc[sample_start:sample_end].reset_index(drop=True)
        if len(original_real_drift_data_chunk) > len(simple_data):
            original_real_drift_data_chunk = original_real_drift_data_chunk[:len(simple_data)]
        original_real_drift_data_chunk[config["applicable_columns"]] = simple_data

        output_filename = f"{args.table_name}.drifted.chunk_{sample_start}_{sample_end}.csv"
        original_real_drift_data_chunk.to_csv(
            os.path.join(save_dir, output_filename), index=False, doublequote=False, escapechar="\\"
        )
        print(f"[Chunk] Saved CSV (no freq preservation) to {output_filename}")

        total_end = time.time()
        print(f"[TIMING] Chunk generation took: {total_end - total_start:.2f}s")
        return

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

    # === BEFORE Frequency Preservation: measure drift and correlation ===
    from calc_drift import calc_drift as measure_drift, calc_correlation
    applicable_columns = config["applicable_columns"]

    print("\n[BEFORE Freq Preservation] Measuring drift and correlation...")
    before_df_partial = data_wrapper.Reverse(sample_data, skip_freq_preservation=True)
    before_df_partial = before_df_partial[applicable_columns]

    # Build full before_df with all columns (for correlation measurement)
    before_df_full = original_real_drift_data.copy()
    if len(before_df_full) > len(before_df_partial):
        before_df_full = before_df_full.iloc[:len(before_df_partial)].reset_index(drop=True)
    elif len(before_df_full) < len(before_df_partial):
        repeat = (len(before_df_partial) // len(before_df_full)) + 1
        before_df_full = pd.concat([before_df_full] * repeat, ignore_index=True).iloc[:len(before_df_partial)]
    before_df_full[applicable_columns] = before_df_partial[applicable_columns].values

    before_drift = measure_drift(train_data, before_df_full, applicable_columns, verbose=False)
    # Use original_data (all columns except 'id') for correlation
    before_corr = calc_correlation(original_data, before_df_full, verbose=False)
    print(f"  BEFORE: drift={before_drift:.6f} (target={args.drift:.6f}, diff={abs(before_drift - args.drift):.6f})")
    before_loss = before_corr.get('pearson', 0)
    before_base_abs = before_corr.get('pearson_abs', 0)
    before_gen_abs = before_corr.get('pearson_abs_gen', 0)
    before_ratio = before_loss / before_base_abs if before_base_abs > 0 else 0
    print(f"  BEFORE: corr base={before_base_abs:.4f}, gen={before_gen_abs:.4f}, loss={before_loss:.4f}, ratio={before_ratio:.4f}")

    # === Set up Frequency Preservation ===
    if args.skip_freq_preservation:
        print("\n[Frequency Preservation] SKIPPED (--skip-freq-preservation)")
        sample_data = data_wrapper.Reverse(sample_data, skip_freq_preservation=True)
        sample_data = sample_data[applicable_columns]
    else:
        print("\n[Setting up Frequency Preservation]")
        for col in applicable_columns:
            if col in data_wrapper.num_normalizer:
                # Only use frequency preservation for foreign key columns
                if col in FK_TO_TABLE:
                    base_col_data = train_data[col].dropna().values
                    if has_real_reference:
                        # Have reference data: detect trend and generate synthetic distribution
                        ref_col_data = real_data[col].dropna().values
                        print(f"  {col}: FK column, using trend-based synthetic distribution "
                              f"(base={len(base_col_data)}, ref={len(ref_col_data)} values)")
                        data_wrapper.set_trend_based_reference_distribution(
                            col, base_col_data, ref_col_data, args.drift)
                    else:
                        # No real reference: use auto mode
                        print(f"  {col}: FK column, synthesizing target distribution with drift={args.drift}")
                        data_wrapper.set_synthetic_reference_distribution(
                            col, base_col_data, args.drift, mode='auto'
                        )
                else:
                    # Non-FK columns (like production_year): skip frequency preservation
                    # Let the model's inverse_transform handle the drift naturally
                    print(f"  {col}: not a FK column, skip frequency preservation")

        sample_data = data_wrapper.Reverse(sample_data)
        sample_data = sample_data[applicable_columns]

    # === AFTER Frequency Preservation: measure drift and correlation ===
    # Build full after_df with all columns (for correlation measurement)
    after_df_full = original_real_drift_data.copy()
    if len(after_df_full) > len(sample_data):
        after_df_full = after_df_full.iloc[:len(sample_data)].reset_index(drop=True)
    elif len(after_df_full) < len(sample_data):
        repeat = (len(sample_data) // len(after_df_full)) + 1
        after_df_full = pd.concat([after_df_full] * repeat, ignore_index=True).iloc[:len(sample_data)]
    after_df_full[applicable_columns] = sample_data[applicable_columns].values

    print("\n[AFTER Freq Preservation] Measuring drift and correlation...")
    after_drift = measure_drift(train_data, after_df_full, applicable_columns, verbose=False)
    # Use original_data (all columns except 'id') for correlation
    after_corr = calc_correlation(original_data, after_df_full, verbose=False)
    print(f"  AFTER:  drift={after_drift:.6f} (target={args.drift:.6f}, diff={abs(after_drift - args.drift):.6f})")
    after_loss = after_corr.get('pearson', 0)
    after_base_abs = after_corr.get('pearson_abs', 0)
    after_gen_abs = after_corr.get('pearson_abs_gen', 0)
    after_ratio = after_loss / after_base_abs if after_base_abs > 0 else 0
    print(f"  AFTER:  corr base={after_base_abs:.4f}, gen={after_gen_abs:.4f}, loss={after_loss:.4f}, ratio={after_ratio:.4f}")
    print(f"  CHANGE: drift {before_drift:.4f} -> {after_drift:.4f}, corr_loss {before_loss:.4f} -> {after_loss:.4f}")

    # if len(original_data.index) > len(sample_data.index):
    # original_data = original_data[:len(sample_data.index)]

    # For chunked generation, slice the reference data to the correct range
    if is_chunked or sample_start > 0:
        original_real_drift_data = original_real_drift_data.iloc[sample_start:sample_end].reset_index(drop=True)

    if len(original_real_drift_data.index) > len(sample_data.index):
        original_real_drift_data = original_real_drift_data[: len(sample_data.index)]

    print("Drifted columns")
    print(sample_data)

    # original_data[config["applicable_columns"]] = sample_data
    original_real_drift_data[config["applicable_columns"]] = sample_data
    print("Drifted data")
    print(original_real_drift_data)

    # Save output (with chunk suffix if chunked generation)
    if is_chunked or sample_start > 0:
        output_filename = f"{args.table_name}.drifted.chunk_{sample_start}_{sample_end}.csv"
    else:
        output_filename = f"{args.table_name}.drifted.csv"

    original_real_drift_data.to_csv(
        os.path.join(save_dir, output_filename),
        index=False,
        doublequote=False,
        escapechar="\\",
    )
    print(f"[Output] Saved to {output_filename}")

    total_end = time.time()
    timing["total"] = total_end - total_start

    # Print timing summary in parseable format
    print("=" * 80)
    print("[TIMING SUMMARY]")
    print(f"  Diffuser Training:   {timing['diffuser_train']:.2f} seconds")
    print(f"  Controller Training: {timing['controller_train']:.2f} seconds")
    print(f"  Training Total:      {timing['diffuser_train'] + timing['controller_train']:.2f} seconds")
    print(f"  Generation:          {timing['generation']:.2f} seconds")
    print(f"  Total:               {timing['total']:.2f} seconds")
    print("=" * 80)
    print(f"[TIMING] Data generation pipeline completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")


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
    parser.add_argument("--sample-steps", type=int, default=None,
                        help="Number of sampling steps (default: same as diffuser-timesteps). Use fewer for faster DDIM-style sampling.")

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
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="Start index for sample generation (for chunked generation)"
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=-1,
        help="Number of samples to generate (-1 for all remaining)"
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
    parser.add_argument("--loss-weight-drift", type=float, default=1.0,
                        help="Weight for drift loss (default: 1.0)")
    parser.add_argument("--loss-weight-corr", type=float, default=0.8,
                        help="Weight for correlation loss (default: 0.8)")
    parser.add_argument("--loss-weight-real", type=float, default=0.1,
                        help="Weight for RealMSE loss (default: 0.1)")
    parser.add_argument("--skip-freq-preservation", action="store_true", default=False,
                        help="Skip frequency preservation (for non-drift-ref mode)")
    parser.add_argument("--train-only", action="store_true", default=False,
                        help="Only train diffuser and controller, skip generation")

    # parser.add_argument("--cond", action="store_true", default=False)

    args = parser.parse_args()

    main(args)
