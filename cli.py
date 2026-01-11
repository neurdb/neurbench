import json
import os
import sys
import shutil
import time
import subprocess
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import button_dialog

from auto_tune import AutoTuner, list_cached_params, get_cached_params_for_table, TuningParams

# Global configuration for all components
GLOBAL_CONFIG = {
    "dataset": "imdb",
    "drift": 0.0,
    "query_set": None,  # Optional: if None, uses queries/{dataset}/train|test
                        # if set, uses queries/{query_set}/train|test
    "pg_port": 5430     # PostgreSQL port (default: 5430)
}

# Tolerance settings for cache matching and validation
CACHE_KEY_TOLERANCE = 0.2       # Max |cache_key - user_drift| for cache key matching
DRIFT_ERROR_TOLERANCE = 0.20    # Max drift_error (|actual - target| / target) for cache quality
CORRELATION_TOLERANCE = 0.25    # Max correlation_loss for cache quality


def get_dataset_dir(dataset_name: str, reference_dataset: str = None) -> str:
    """Get dataset directory name with reference_dataset suffix."""
    if reference_dataset and reference_dataset != dataset_name:
        return f"{dataset_name}_ref_{reference_dataset}"
    return dataset_name


def get_expdir_path(dataset_name: str, table_name: str, reference_dataset: str = None, variant_id: int = -1) -> str:
    """Get full expdir path for a table."""
    dataset_dir = get_dataset_dir(dataset_name, reference_dataset)
    if variant_id > 0:
        dataset_dir += f"-{variant_id}"
    return os.path.join("expdir", dataset_dir, table_name)


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def print_args(**kwargs):
    print("-" * 20)
    for key, value in kwargs.items():
        print(f"{key}: {value}")
    print("-" * 20)


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Smart CSV loader that auto-detects the correct quoting strategy.
    If first strategy has warnings/skipped lines, try next one.
    """
    import warnings

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


# class CommandPrompter(Validator):
#     def validate(self, document):
#         text = document.text

#         if text and not text.isdigit():
#             i = 0

#             # Get index of first non numeric character.
#             # We want to move the cursor here.
#             for i, c in enumerate(text):
#                 if not c.isdigit():
#                     break

#             raise ValidationError(
#                 message="This input contains non-numeric characters", cursor_position=i
#             )


print(
    "Welcome to NRBench interactive shell. Type 'h' to view a list of commands. Type 'q' to quit."
)


def show_help():
    print(
        """NRBench - Available Commands
"======================================================"

Core Commands:
  h, help                            Show this help message
  q, quit                            Exit the interactive shell
  set [KEY] [VALUE]                  Set global configuration parameters
                                       - dataset: imdb, books, fb, osm, wiki
                                       - drift: 0.0-1.0
                                       - query_set: query set name (e.g., join-order-benchmark)
  set                                Show current configuration
  gd DATASET [TABLE] DRIFT [--auto] [--gpus=0,1,2] [--validate] [--ops=MODE]
                                     Generate data that drifts DRIFT on DATASET
                                       --auto: Auto-tune parameters for target drift
                                       --gpus=X,Y,Z: Use multiple GPUs for parallel execution
                                                     (only when TABLE is not specified)
                                       --validate: Validate with DB after generation, re-tune if needed
                                                   (pass if ratio in [1/1.2, 1.2] = within ±20%)
                                       --dry-run: Test validation without replacing tables
                                       --ops=MODE: all|retain|regene-only (default: retain)
  gd DATASET --drift-ref=FILE [--gpus=0,1,2] [--validate] [--dry-run] [--ops=MODE] [--batch-size=N]
                                     Generate data using drift reference file
                                       The reference file should contain per-table drift values
                                       (e.g., from calc_drift.py --src-dir X --dst-dir Y)
                                       --ops=MODE: Operation mode:
                                         - all: Full auto-tune from scratch (ignore cache)
                                         - retain: Use cache if available, auto-tune if not (default)
                                         - regene-only: Only regenerate using cached params, skip training
                                       --batch-size=N: Samples per batch (default 524288, larger = fewer inits)
  gq DATASET DRIFT                   Generate query that drifts DRIFT on DATASET
  dd DATASET [TABLE]                 Delete data generator model for DATASET
  dq DATASET                         Delete query generator model for DATASET
  tqo [LQO_NAME]                     Train learned query optimizer
                                       Uses queries/{query_set}/train or queries/{dataset}/train
  iqo [LQO_NAME] [MODE]              Test learned query optimizer
                                       Uses queries/{query_set}/test or queries/{dataset}/test
                                     For bao: iqo bao [bao|pg]
                                       - bao: Test with Bao optimizer (default)
                                       - pg:  Test with PostgreSQL optimizer
  idx [INDEX_NAME]                   Test learned index
  lcc                                Test learned concurrency control
  vd DATASET [TABLE] [--gen-db=X] [--real-db=Y]
                                     Validate generated data against real database
                                       - Imports generated tables into gen-db
                                       - Runs JOB queries on both databases
                                       - Compares execution times per table
                                       --gen-db: Target database for generated data (default: imdb_17v2_gen)
                                       --real-db: Reference real database (default: imdb_17v2)

Configuration Tips:
  set dataset imdb_ori               # Set database to use
  set query_set join-order-benchmark # Set query set (overrides dataset queries)
  set query_set none                 # Use default queries/{dataset}/train|test

"""
    )


def _run_table_generation(
    dataset_name: str,
    table_name: str,
    drift: float,
    scale: float,
    auto_tune: bool,
    reference_dataset: Optional[str],
    device: int,
) -> dict:
    """
    Run data generation for a single table. This function is designed to be
    called in a subprocess for parallel execution.

    Returns a dict with status information.
    """
    import sys
    import os

    result = {
        "table_name": table_name,
        "device": device,
        "success": False,
        "message": "",
    }

    try:
        ref_arg = f" --reference-dataset={reference_dataset}" if reference_dataset else ""

        if auto_tune:
            # For auto-tune, we need to run as subprocess to capture output
            from auto_tune import AutoTuner

            tuner = AutoTuner(
                dataset_name=dataset_name,
                table_name=table_name,
                reference_dataset=reference_dataset,
                device=device,
                verbose=True,
            )

            tune_result = tuner.tune(drift, max_iterations=50, tolerance=DRIFT_ERROR_TOLERANCE)

            if tune_result:
                result["success"] = True
                result["message"] = f"drift_error={tune_result.drift_error:.4f}, corr_loss={tune_result.correlation_loss:.4f}"
            else:
                result["message"] = "Auto-tuning failed"
        else:
            # Check for cached params first
            cached_params = get_cached_params_for_table(dataset_name, table_name, drift, reference_dataset)
            if cached_params:
                cmd = f"python3 -u dbproc.py --dataset-name={dataset_name} --table-name={table_name} --drift={drift}"
                cmd += f" --device={device} {cached_params.to_cmd_args()} --reuse{ref_arg}"
            else:
                cmd = f"python3 -u dbproc.py --dataset-name={dataset_name} --table-name={table_name} --drift={drift}"
                cmd += f" --device={device} --scale-factor={scale}{ref_arg}"

            ret = os.system(cmd)
            result["success"] = (ret == 0)
            result["message"] = "completed" if ret == 0 else f"exit code {ret}"

    except Exception as e:
        result["message"] = str(e)

    return result


def load_drift_reference(filepath: str, src_dataset: str = None, dst_dataset: str = None) -> dict:
    """
    Load drift reference file (CSV) and return {table_name: (drift, corr_loss)}.

    If src_dataset and dst_dataset are provided, only return rows matching both.
    This allows a single CSV to contain drift values for multiple dataset pairs.

    Returns dict with table_name -> (drift, pearson_corr_loss) tuples.

    Expected CSV columns:
    - table_name: name of the table
    - drift: target drift value
    - pearson_corr_loss: (optional) target correlation loss
    - src_dataset: (optional) source dataset used in calc_drift (e.g., imdb)
    - dst_dataset: (optional) destination dataset used in calc_drift (e.g., imdb_2017)
    """
    import pandas as pd
    if not os.path.exists(filepath):
        print(f"Error: Drift reference file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        # Expected columns: table_name, drift, pearson_corr_loss, src_dataset, dst_dataset
        if "table_name" not in df.columns or "drift" not in df.columns:
            print("Error: Drift reference file must have 'table_name' and 'drift' columns")
            return None
        # Filter out rows with null drift
        df = df[df["drift"].notna()]

        # Filter by src_dataset and dst_dataset if provided
        has_src_col = "src_dataset" in df.columns
        has_dst_col = "dst_dataset" in df.columns

        if src_dataset and has_src_col:
            df = df[df["src_dataset"] == src_dataset]
        if dst_dataset and has_dst_col:
            df = df[df["dst_dataset"] == dst_dataset]

        if len(df) == 0:
            print(f"Error: No matching rows found for src_dataset={src_dataset}, dst_dataset={dst_dataset}")
            return None

        result = {}
        for _, row in df.iterrows():
            table_name = row["table_name"]
            drift = row["drift"]
            # Get pearson correlation loss if available
            corr_loss = None
            if "pearson_corr_loss" in df.columns and pd.notna(row.get("pearson_corr_loss")):
                corr_loss = row["pearson_corr_loss"]
            result[table_name] = (drift, corr_loss)
        return result
    except Exception as e:
        print(f"Error loading drift reference file: {e}")
        return None


def _validate_generation_results(
    dataset_name: str,
    tables: List[str],
    drift_ref: dict,
    reference_dataset: str = None,
):
    """Validate generated data against original and compare with target."""
    import calc_drift

    print(f"\n{'='*70}")
    print("VALIDATION: Generated vs Original (compared with Target)")
    print(f"{'='*70}")

    base_dir = os.path.join("datasets", dataset_name)
    config = load_json(os.path.join(base_dir, "dataset_info.json"))

    for t in tables:
        table_config = config.get(t, {})
        if not table_config:
            continue
        columns = table_config.get("applicable_columns", [])
        target_drift, target_corr = drift_ref.get(t, (None, None))

        orig_path = os.path.join(base_dir, f"{t}.csv")
        gen_path = os.path.join(get_expdir_path(dataset_name, t, reference_dataset), f"{t}.drifted.csv")

        if not os.path.exists(gen_path):
            print(f"  {t}: generated file not found")
            continue

        try:
            orig_df = calc_drift.load_csv(orig_path)
            gen_df = calc_drift.load_csv(gen_path)

            actual_drift = calc_drift.calc_drift(orig_df, gen_df, columns, verbose=False)
            corr_results = calc_drift.calc_correlation(orig_df, gen_df, verbose=False)
            actual_corr = corr_results.get("pearson", 0.0)

            # Format output
            drift_str = f"drift={actual_drift:.4f}"
            if target_drift:
                drift_str += f" (target={target_drift:.4f}, Δ={actual_drift - target_drift:+.4f})"
            corr_str = f"corr={actual_corr:.4f}"
            if target_corr:
                corr_str += f" (target={target_corr:.4f}, Δ={actual_corr - target_corr:+.4f})"
            print(f"  {t}: {drift_str}, {corr_str}")

        except Exception as e:
            print(f"  {t}: error - {e}")

    print(f"{'='*70}\n")


def _merge_npy_with_freq_preservation(
    dataset_name: str, table_name: str, npy_files: List[str],
    reference_dataset: Optional[str], drift: float
) -> Optional[pd.DataFrame]:
    """Merge .npy chunks and apply unified Frequency Preservation."""
    import pickle
    import numpy as np
    from calc_drift import calc_drift as measure_drift, calc_correlation

    try:
        save_dir = os.path.dirname(npy_files[0])
        base_dir = os.path.join("datasets", dataset_name)

        # 1. Load and concat all .npy files
        all_data = [np.load(f) for f in npy_files]
        merged_data = np.concatenate(all_data, axis=0)
        print(f"    Loaded {len(npy_files)} chunks -> {len(merged_data)} rows")

        # 2. Load data_wrapper and config
        with open(os.path.join(save_dir, "data_wrapper.pkl"), "rb") as f:
            data_wrapper = pickle.load(f)
        config = load_json(os.path.join(base_dir, "dataset_info.json")).get(table_name, {})
        applicable_columns = config.get("applicable_columns", [])

        # 3. Set up reference distributions for FK columns
        FK_TO_TABLE = {
            'movie_id': 'title', 'person_id': 'name', 'company_id': 'company_name',
            'keyword_id': 'keyword', 'linked_movie_id': 'title', 'link_type_id': 'link_type',
            'info_type_id': 'info_type', 'kind_id': 'kind_type', 'role_id': 'role_type',
        }
        has_real_ref = reference_dataset and reference_dataset != dataset_name

        # Load reference data if available
        real_data = None
        if has_real_ref:
            real_path = os.path.join("datasets", reference_dataset, f"{table_name}.csv")
            if os.path.exists(real_path):
                try:
                    real_data = load_csv(real_path)
                except RuntimeError:
                    pass

        # Load train data (original/base data) for comparison
        train_data = None
        train_path = os.path.join(base_dir, f"{table_name}.csv")
        try:
            train_data = load_csv(train_path)
        except RuntimeError:
            pass

        # === BEFORE Frequency Preservation: measure drift and correlation ===
        print(f"\n    [BEFORE Freq Preservation] Measuring drift and correlation...")
        before_df_partial = data_wrapper.Reverse(merged_data, skip_freq_preservation=True)
        before_df_partial = before_df_partial[applicable_columns]

        # Build full before_df with all columns (for correlation measurement)
        before_ref_path = os.path.join("datasets", reference_dataset or dataset_name, f"{table_name}.csv")
        before_ref_df = None
        try:
            before_ref_df = load_csv(before_ref_path)
        except RuntimeError:
            pass

        if before_ref_df is not None:
            if len(before_ref_df) > len(before_df_partial):
                before_ref_df = before_ref_df.iloc[:len(before_df_partial)].reset_index(drop=True)
            elif len(before_ref_df) < len(before_df_partial):
                repeat = (len(before_df_partial) // len(before_ref_df)) + 1
                before_ref_df = pd.concat([before_ref_df] * repeat, ignore_index=True).iloc[:len(before_df_partial)]
            before_ref_df[applicable_columns] = before_df_partial[applicable_columns].values
            before_df = before_ref_df
        else:
            before_df = before_df_partial

        before_drift, before_corr = None, None
        before_loss, before_base_abs, before_gen_abs = 0, 0, 0
        if train_data is not None:
            before_drift = measure_drift(train_data, before_df, applicable_columns, verbose=False)
            before_corr = calc_correlation(train_data, before_df, verbose=False)
            print(f"    BEFORE: drift={before_drift:.6f} (target={drift:.6f}, diff={abs(before_drift - drift):.6f})")
            before_loss = before_corr.get('pearson', 0)
            before_base_abs = before_corr.get('pearson_abs', 0)
            before_gen_abs = before_corr.get('pearson_abs_gen', 0)
            before_ratio = before_loss / before_base_abs if before_base_abs > 0 else 0
            print(f"    BEFORE: corr base={before_base_abs:.4f}, gen={before_gen_abs:.4f}, loss={before_loss:.4f}, ratio={before_ratio:.4f}")

        # === Set up Frequency Preservation ===
        print(f"\n    Setting up Frequency Preservation (trend-based)...")
        for col in applicable_columns:
            if col in data_wrapper.num_normalizer and col in FK_TO_TABLE:
                if train_data is None or col not in train_data.columns:
                    continue
                base_col_data = train_data[col].dropna().values

                if has_real_ref and real_data is not None and col in real_data.columns:
                    # Have reference data: detect trend and generate synthetic distribution
                    ref_col_data = real_data[col].dropna().values
                    data_wrapper.set_trend_based_reference_distribution(
                        col, base_col_data, ref_col_data, drift)
                else:
                    # No reference data: use auto mode
                    data_wrapper.set_synthetic_reference_distribution(
                        col, base_col_data, drift, mode='auto')

        # 4. Apply Reverse with frequency preservation
        sample_df = data_wrapper.Reverse(merged_data)
        sample_df = sample_df[applicable_columns]

        # 5. Load reference data for non-applicable columns
        ref_path = os.path.join("datasets", reference_dataset or dataset_name, f"{table_name}.csv")
        ref_df = None
        try:
            ref_df = load_csv(ref_path)
        except RuntimeError:
            pass

        # Build final df with all columns
        if ref_df is not None:
            # Adjust ref_df size to match sample_df
            if len(ref_df) > len(sample_df):
                ref_df = ref_df.iloc[:len(sample_df)].reset_index(drop=True)
            elif len(ref_df) < len(sample_df):
                repeat = (len(sample_df) // len(ref_df)) + 1
                ref_df = pd.concat([ref_df] * repeat, ignore_index=True).iloc[:len(sample_df)]
            ref_df[applicable_columns] = sample_df[applicable_columns].values
            final_df = ref_df
        else:
            final_df = sample_df

        # === AFTER Frequency Preservation: measure drift and correlation ===
        # Use full data (all columns) to match calc_drift.py and auto_tune.py
        print(f"\n    [AFTER Freq Preservation] Measuring drift and correlation...")
        if train_data is not None:
            after_drift = measure_drift(train_data, final_df, applicable_columns, verbose=False)
            after_corr = calc_correlation(train_data, final_df, verbose=False)
            print(f"    AFTER:  drift={after_drift:.6f} (target={drift:.6f}, diff={abs(after_drift - drift):.6f})")
            after_loss = after_corr.get('pearson', 0)
            after_base_abs = after_corr.get('pearson_abs', 0)
            after_gen_abs = after_corr.get('pearson_abs_gen', 0)
            after_ratio = after_loss / after_base_abs if after_base_abs > 0 else 0
            print(f"    AFTER:  corr base={after_base_abs:.4f}, gen={after_gen_abs:.4f}, loss={after_loss:.4f}, ratio={after_ratio:.4f}")
            if before_drift is not None and before_corr is not None:
                print(f"    CHANGE: drift {before_drift:.4f} -> {after_drift:.4f}, corr_loss {before_loss:.4f} -> {after_loss:.4f}")

        return final_df

    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def handle_generate_data(tokens: List[str]):
    tokens = tokens[1:]

    if len(tokens) < 1:
        print("Error: Not enough arguments")
        print("Usage: gd DATASET [TABLE] DRIFT [--auto] [--ref=REFERENCE_DATASET] [--gpus=0,1,2]")
        print("       gd DATASET --drift-ref=FILE [--gpus=0,1,2]")
        print("  --auto: Enable auto-tuning to find best parameters")
        print("  --ref=X: Use dataset X as reference for drift direction (default: DATASET_2014)")
        print("  --gpus=X,Y,Z: Use multiple GPUs for parallel execution (when no TABLE specified)")
        print("  --drift-ref=FILE: Use drift reference file (per-table target drift values)")
        print("Example: gd imdb title 0.3 --ref=imdb_2017")
        print("Example: gd imdb 0.3 --auto --gpus=0,1,2,3")
        print("Example: gd imdb --drift-ref=drift_reference.csv --gpus=0,1,2,3")
        return

    dataset_name = tokens[0]
    table_name = ""
    drift = 0.0
    scale = 1.0
    auto_tune = False
    reference_dataset = None
    gpus = None  # List of GPU IDs for parallel execution
    drift_ref_file = None  # Drift reference file path
    validate_after = False  # Whether to validate with DB after generation
    validate_threshold = 1.2  # 20% performance threshold
    dry_run = False  # Dry run: test without replacing tables
    ops_mode = "retain"  # Operation mode: all, retain, regene-only
    exclude_tables = []  # Tables to exclude from generation
    batch_size = 524288  # Samples per batch (default 512K)
    sample_steps = None  # DDIM sampling steps (None = use diffuser-timesteps)
    variant_id = -1  # Variant ID for creating separate output directories

    # Parse flags
    remaining_tokens = []
    for t in tokens[1:]:
        if t == "--auto":
            auto_tune = True
        elif t == "--validate":
            validate_after = True
        elif t == "--dry-run":
            dry_run = True
            validate_after = True  # dry-run implies validate
        elif t == "--force":
            # Legacy flag: --force is equivalent to --ops=regene-only
            ops_mode = "regene-only"
        elif t.startswith("--ops="):
            ops_mode = t[6:]  # Extract value after --ops=
            if ops_mode not in ("all", "retain", "retrain", "retrain-only", "regene-only"):
                print(f"Error: Invalid --ops value '{ops_mode}'. Must be: all, retain, retrain, retrain-only, regene-only")
                return
        elif t.startswith("--validate-threshold="):
            validate_threshold = float(t[21:])
            validate_after = True
        elif t.startswith("--ref="):
            reference_dataset = t[6:]  # Extract value after --ref=
        elif t.startswith("--gpus="):
            gpus_str = t[7:]  # Extract value after --gpus=
            gpus = [int(g.strip()) for g in gpus_str.split(",")]
        elif t.startswith("--drift-ref="):
            drift_ref_file = t[12:]  # Extract value after --drift-ref=
            auto_tune = True  # drift-ref implies auto-tune
        elif t.startswith("--exclude="):
            exclude_tables = [x.strip() for x in t[10:].split(",")]
        elif t.startswith("--batch-size="):
            batch_size = int(t[13:])
        elif t.startswith("--sample-steps="):
            sample_steps = int(t[15:])
        elif t.startswith("--variant-id="):
            variant_id = int(t[13:])
        else:
            remaining_tokens.append(t)

    tokens = remaining_tokens

    # Check if a table name is specified (first remaining token that's not a number)
    specified_table = None
    if tokens and not is_float(tokens[0]) and not is_integer(tokens[0]):
        specified_table = tokens[0]
        tokens = tokens[1:]

    # If using drift reference file, handle separately
    if drift_ref_file:
        _generate_with_drift_reference(
            dataset_name, drift_ref_file, reference_dataset, gpus,
            single_table=specified_table,
            validate_after=validate_after,
            validate_threshold=validate_threshold,
            dry_run=dry_run,
            ops_mode=ops_mode,
            exclude_tables=exclude_tables,
            batch_size=batch_size,
            sample_steps=sample_steps,
            variant_id=variant_id,
        )
        return

    if len(tokens) < 1:
        print("Error: Not enough arguments (need DRIFT value or --drift-ref=FILE)")
        return

    # Use specified_table if we already parsed it, otherwise parse from tokens
    if specified_table:
        table_name = specified_table
        if not is_float(tokens[0]) and not is_integer(tokens[0]):
            print("Invalid drift value:", tokens[0])
            return
        drift = float(tokens[0])
        if len(tokens) > 1:
            scale = float(tokens[1])
    elif is_float(tokens[0]) or is_integer(tokens[0]):
        drift = float(tokens[0])
        if len(tokens) > 1:
            scale = float(tokens[1])
    else:
        if len(tokens) < 2:
            print("Error: Not enough arguments")
            return

        table_name = tokens[0]
        if not is_float(tokens[1]) and not is_integer(tokens[1]):
            print("Invalid drift value:", tokens[1])
            return
        else:
            drift = float(tokens[1])
            if len(tokens) > 2:
                scale = float(tokens[2])

    # Build reference dataset argument
    ref_arg = f" --reference-dataset={reference_dataset}" if reference_dataset else ""

    if table_name:
        device = gpus[0] if gpus else 0

        # Check cache status for display and skip logic
        cache_status = _check_cache_status(dataset_name, table_name, drift, reference_dataset)

        print_args(
            dataset_name=dataset_name,
            table_name=table_name,
            drift=drift,
            reference_dataset=reference_dataset or f"{dataset_name}_2014 (default)",
            device=device,
            cache_status="validated" if cache_status["validation_passed"] else
                        ("cached" if cache_status["has_cache"] else "no cache"),
        )

        # Check if already validated and should skip (only for retain mode)
        if cache_status["validation_passed"] and ops_mode == "retain":
            print(f"\nAlready validated (validation_passed=True), skipping generation.")
            print(f"Use --ops=regene-only to regenerate, or --ops=all to retrain from scratch.")
        elif ops_mode == "retrain":
            # retrain mode: use cached/default params and force retrain (no tuning)
            print(f"\n[ops=retrain] Force retrain with cached/default params (no tuning)...")
            tuner = AutoTuner(
                dataset_name=dataset_name,
                table_name=table_name,
                reference_dataset=reference_dataset,
                device=device,
                verbose=True,
                sample_steps=sample_steps,
                skip_freq_preservation=False,  # Use synthetic distribution for freq preservation
            )
            success, output_path = tuner.generate_with_best_params(drift, use_cache=True, retrain=True)
            if success:
                print(f"Retrain completed: {output_path}")
            else:
                print(f"Retrain failed!")
        else:
            # Map ops_mode to internal flags:
            # - all: force retrain from scratch (no_cache=True)
            # - retain: use cache if available (default)
            # - regene-only: only regenerate with cached params (force_regenerate=True)
            force_regenerate = (ops_mode == "regene-only")
            no_cache = (ops_mode == "all")
            _generate_with_auto_tune(
                dataset_name, table_name, drift, reference_dataset, device,
                force_regenerate=force_regenerate,
                no_cache=no_cache,
                require_validation=False,
                sample_steps=sample_steps,
                skip_freq_preservation=False,  # Use synthetic distribution for freq preservation
            )

        # Run validation if requested
        if validate_after:
            print(f"\n{'#'*60}")
            print(f"VALIDATION PHASE")
            print(f"{'#'*60}")
            failed_tables, import_failed_tables = _validate_and_get_failed_tables(
                dataset_name, [table_name], validate_threshold,
                reference_dataset=reference_dataset,
                variant_id=variant_id,
                dry_run=dry_run,
                skip_freq_preservation=False,  # Use synthetic distribution for freq preservation
            )
            if not failed_tables and not import_failed_tables:
                print(f"\n✓ Table {table_name} passed validation!")
            elif import_failed_tables:
                print(f"\n✗ Table {table_name} import failed - check CSV format")
            else:
                print(f"\n✗ Table {table_name} failed validation")
    else:
        # run on all tables
        ## get table names from dataset_info.json
        base_dir = os.path.join("datasets", dataset_name)
        config: dict = load_json(os.path.join(base_dir, "dataset_info.json"))
        table_names = [t for t in config.keys() if config[t]]

        # Filter out excluded tables
        if exclude_tables:
            table_names = [t for t in table_names if t not in exclude_tables]
            print(f"Excluding tables: {exclude_tables}")

        # Check cache status for each table and classify
        tables_to_skip = set()        # Tables to skip (validation passed, ops=retain)
        tables_generate_only = set()  # Tables with cache (generate only, no retrain)
        tables_to_generate = {}       # {table: drift} - all tables to process

        for t in table_names:
            cache_status = _check_cache_status(dataset_name, t, drift, reference_dataset)
            if ops_mode == "all":
                # Force retrain all
                tables_to_generate[t] = drift
            elif cache_status["validation_passed"]:
                if ops_mode == "retain":
                    tables_to_skip.add(t)
                else:  # regene-only
                    tables_to_generate[t] = drift
                    tables_generate_only.add(t)
            elif cache_status["has_cache"]:
                tables_to_generate[t] = drift
                tables_generate_only.add(t)
            else:
                tables_to_generate[t] = drift

        # Print header
        print(f"\n{'='*60}")
        print(f"GENERATION MODE")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Target drift: {drift}")
        print(f"Reference dataset: {reference_dataset or '(default)'}")
        print(f"Tables to generate: {list(tables_to_generate.keys())}")
        print(f"GPUs: {gpus if gpus else [0]}")
        print(f"Operation mode: {ops_mode}")
        print()

        # Print per-table status
        print("Target per table:")
        for t in table_names:
            cache_status = _check_cache_status(dataset_name, t, drift, reference_dataset)
            if ops_mode == "all":
                status_str = " [RETRAIN: --ops=all]"
            elif t in tables_to_skip:
                status_str = " [CACHE: validated, SKIP]"
            elif cache_status["validation_passed"]:
                status_str = " [CACHE: validated, REGEN]"
            elif cache_status["has_cache"]:
                status_str = " [CACHE: OK, generate only]"
            else:
                status_str = " [NO CACHE: auto-tune]"
            print(f"  {t}: drift={drift:.4f}{status_str}")
        print()

        if tables_to_skip:
            print(f"Skipping {len(tables_to_skip)} validated table(s): {list(tables_to_skip)}")
            print()

        if not tables_to_generate:
            print("All tables already validated. Nothing to generate.")
            print("Use --ops=regene-only to regenerate, or --ops=all to retrain.")
            return

        # Start timing for generation
        generation_start_time = time.time()

        if gpus and len(gpus) > 1:
            # Multi-GPU parallel execution with two-phase structure
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import subprocess

            log_subdir = get_dataset_dir(dataset_name, reference_dataset)
            if variant_id > 0:
                log_subdir += f"-{variant_id}"
            log_dir = os.path.join("gd_logs", log_subdir)
            os.makedirs(log_dir, exist_ok=True)
            generated_tables = set()
            force_retrain_tables = set()

            # Separate tables: those needing auto-tune vs those with cached params
            tables_need_autotune = {t: d for t, d in tables_to_generate.items()
                                    if t not in tables_generate_only}
            tables_cached = {t: d for t, d in tables_to_generate.items()
                             if t in tables_generate_only}

            # Phase 1: Auto-tune for tables without cache
            if tables_need_autotune:
                print(f"=== Phase 1: Auto-tune ({len(tables_need_autotune)} tables) ===")
                print("These tables need auto-tune and will generate full tables directly.")
                print()

                def run_autotune(table_name, target_drift, gpu_id):
                    cmd = f"python3 -u auto_tune.py --dataset-name={dataset_name} --table-name={table_name}"
                    cmd += f" --target-drift={target_drift} --device={gpu_id}"
                    if reference_dataset:
                        cmd += f" --reference-dataset={reference_dataset}"
                    if sample_steps:
                        cmd += f" --sample-steps={sample_steps}"
                    if variant_id > 0:
                        cmd += f" --variant-id={variant_id}"
                    # Map ops_mode to auto_tune.py flags
                    if ops_mode == "all" or table_name in force_retrain_tables:
                        cmd += " --no-cache"
                    elif ops_mode == "regene-only":
                        cmd += " --force-regenerate"

                    log_file = os.path.join(log_dir, f"{table_name}_{target_drift}.log")
                    print(f"[GPU {gpu_id}] Auto-tune: {table_name} (log: {log_file})")

                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    with open(log_file, "w", buffering=1) as f:
                        proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env)
                        ret = proc.wait()

                    success = ret == 0
                    status = "Done" if success else "FAIL"
                    print(f"[GPU {gpu_id}] {status}: {table_name}")
                    return table_name, success

                # Assign tables to GPUs round-robin
                autotune_tasks = list(tables_need_autotune.items())
                failed = []
                with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
                    futures = {}
                    for i, (t, target_drift) in enumerate(autotune_tasks):
                        gpu_id = gpus[i % len(gpus)]
                        futures[executor.submit(run_autotune, t, target_drift, gpu_id)] = t

                    for future in as_completed(futures):
                        table_name, success = future.result()
                        if success:
                            generated_tables.add(table_name)
                            tables_generate_only.add(table_name)  # Now has cache
                        else:
                            failed.append(table_name)

                print(f"=== Phase 1 complete ===\n")

            # Phase 2: Generate with cached params for remaining tables
            if tables_cached:
                print(f"=== Phase 2: Generate with cached params ({len(tables_cached)} tables) ===")
                print("These tables have cached params and will generate directly.")
                print()

                def run_generate(table_name, target_drift, gpu_id):
                    # Use cached params via auto_tune.py with default behavior (uses cache)
                    cmd = f"python3 -u auto_tune.py --dataset-name={dataset_name} --table-name={table_name}"
                    cmd += f" --target-drift={target_drift} --device={gpu_id}"
                    if reference_dataset:
                        cmd += f" --reference-dataset={reference_dataset}"
                    if sample_steps:
                        cmd += f" --sample-steps={sample_steps}"
                    if variant_id > 0:
                        cmd += f" --variant-id={variant_id}"
                    if ops_mode == "regene-only":
                        cmd += " --force-regenerate"

                    log_file = os.path.join(log_dir, f"{table_name}_{target_drift}.log")
                    print(f"[GPU {gpu_id}] Generate: {table_name} (log: {log_file})")

                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    with open(log_file, "w", buffering=1) as f:
                        proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env)
                        ret = proc.wait()

                    success = ret == 0
                    status = "Done" if success else "FAIL"
                    print(f"[GPU {gpu_id}] {status}: {table_name}")
                    return table_name, success

                # Assign tables to GPUs round-robin
                generate_tasks = list(tables_cached.items())
                with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
                    futures = {}
                    for i, (t, target_drift) in enumerate(generate_tasks):
                        gpu_id = gpus[i % len(gpus)]
                        futures[executor.submit(run_generate, t, target_drift, gpu_id)] = t

                    for future in as_completed(futures):
                        table_name, success = future.result()
                        if success:
                            generated_tables.add(table_name)
                        else:
                            failed.append(table_name)

                print(f"=== Phase 2 complete ===\n")

            if failed:
                print(f"Failed tables: {failed}")

        else:
            # Sequential execution with two-phase structure
            device = gpus[0] if gpus else 0

            # Separate tables: those needing auto-tune vs those with cached params
            tables_need_autotune = [t for t in tables_to_generate.keys()
                                    if t not in tables_generate_only]
            tables_cached = [t for t in tables_to_generate.keys()
                             if t in tables_generate_only]

            # Phase 1: Auto-tune for tables without cache
            if tables_need_autotune:
                print(f"=== Phase 1: Auto-tune ({len(tables_need_autotune)} tables) ===")
                for t in tables_need_autotune:
                    print(f"\n{'='*60}")
                    print(f"[Phase 1] Auto-tuning {dataset_name}.{t} on GPU {device}...")
                    print(f"{'='*60}")

                    _generate_with_auto_tune(
                        dataset_name, t, drift, reference_dataset, device,
                        force_regenerate=False,
                        no_cache=(ops_mode == "all"),
                        require_validation=False,
                        sample_steps=sample_steps,
                        skip_freq_preservation=False,  # Use synthetic distribution for freq preservation
                    )
                    print(f"Auto-tune complete for {dataset_name}.{t}")
                print(f"=== Phase 1 complete ===\n")

            # Phase 2: Generate with cached params
            if tables_cached:
                print(f"=== Phase 2: Generate with cached params ({len(tables_cached)} tables) ===")
                for t in tables_cached:
                    print(f"\n{'='*60}")
                    print(f"[Phase 2] Generating {dataset_name}.{t} on GPU {device}...")
                    print(f"{'='*60}")

                    _generate_with_auto_tune(
                        dataset_name, t, drift, reference_dataset, device,
                        force_regenerate=True,
                        no_cache=False,
                        require_validation=False,
                        sample_steps=sample_steps,
                        skip_freq_preservation=False,  # Use synthetic distribution for freq preservation
                    )
                    print(f"Generation complete for {dataset_name}.{t}")
                print(f"=== Phase 2 complete ===\n")

        # Print total time
        generation_total_time = time.time() - generation_start_time
        hours, remainder = divmod(generation_total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\n{'='*60}")
        print(f"All tables processed")
        if hours > 0:
            print(f"Total time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
        elif minutes > 0:
            print(f"Total time: {int(minutes)}m {seconds:.1f}s")
        else:
            print(f"Total time: {seconds:.1f}s")
        print(f"{'='*60}")


def _generate_with_auto_tune(
    dataset_name: str,
    table_name: str,
    drift: float,
    reference_dataset: str = None,
    device: int = 0,
    target_corr_loss: float = None,
    force_regenerate: bool = False,
    no_cache: bool = False,
    require_validation: bool = False,
    sample_steps: int = None,
    skip_freq_preservation: bool = False,
) -> bool:
    """Generate data with automatic parameter tuning.

    Args:
        force_regenerate: If True, use cached params to regenerate (don't skip), train only if no cache.
        no_cache: If True, ignore cache and force retrain (for validation-failed retries).
        skip_freq_preservation: If True, skip frequency preservation (for non-drift-ref mode).

    Returns:
        True if used validated cache (skip validation), False otherwise
    """
    corr_str = f", target_corr={target_corr_loss:.4f}" if target_corr_loss is not None else ""
    if force_regenerate:
        cache_str = " [FORCE REGEN]"
    elif no_cache:
        cache_str = " [NO CACHE]"
    else:
        cache_str = ""
    print(f"\nAuto-tuning parameters for {dataset_name}.{table_name} (drift={drift}{corr_str}) on GPU {device}{cache_str}...")
    if reference_dataset:
        print(f"Reference dataset: {reference_dataset}")

    tuner = AutoTuner(
        dataset_name=dataset_name,
        table_name=table_name,
        reference_dataset=reference_dataset,
        device=device,
        verbose=True,
        target_corr_loss=target_corr_loss,
        sample_steps=sample_steps,
        force_cache_update=no_cache,  # ops=all: always update cache with new result
        skip_freq_preservation=skip_freq_preservation,
    )

    # --force-regenerate: use cached params to regenerate, train only if no cache
    if force_regenerate and not no_cache:
        cached = tuner.cache.get_best_params(drift)
        if cached:
            print(f"Using cached params (drift_error={cached.drift_error:.2%}) to force regenerate...")
            success, output_path = tuner.generate_with_best_params(drift, use_cache=True)
            if success:
                print(f"Force regenerate completed: {output_path}")
                return False  # Don't skip validation
            else:
                print("Force regenerate failed, falling through to tune...")
        else:
            print("No cached params found, will train...")

    result = tuner.tune(drift, max_iterations=50, tolerance=DRIFT_ERROR_TOLERANCE, use_cache=not no_cache, require_validation=require_validation)

    used_validated_cache = False

    if result:
        print(f"\nAuto-tuning complete!")
        print(f"  Target drift: {drift:.4f}")
        print(f"  Actual drift: {result.actual_drift:.4f}")
        print(f"  Drift error: {result.drift_error:.2%}")  # Relative error
        print(f"  Correlation loss: {result.correlation_loss:.4f}")
        if target_corr_loss is not None:
            corr_error = abs(target_corr_loss - result.correlation_loss)  # Absolute error
            print(f"  Target corr loss: {target_corr_loss:.4f}")
            print(f"  Corr error: {corr_error:.4f} (tolerance: {CORRELATION_TOLERANCE})")
        print(f"  Best scale_factor: {result.params.scale_factor}")

        # Check if this was a validated cache hit
        if result.validation_passed:
            used_validated_cache = True
            print(f"  Using validated cache (skip validation)")

        # Ensure we have data generated with best params
        output_path = os.path.join(get_expdir_path(dataset_name, table_name, reference_dataset), f"{table_name}.drifted.csv")
        if not os.path.exists(output_path):
            # File doesn't exist, generate with best cached params
            print(f"\nGenerating data with cached parameters...")
            tuner.generate_with_best_params(drift, use_cache=True)
        else:
            # File exists, check if it was generated with best params
            # (last iteration might not be the best one)
            regenerate_with_best_cached_params(dataset_name, table_name, skip_freq_preservation=skip_freq_preservation, reference_dataset=reference_dataset)
    else:
        print(f"\nAuto-tuning failed. Using default parameters.")
        ref_arg = f" --reference-dataset={reference_dataset}" if reference_dataset else ""
        skip_freq_arg = " --skip-freq-preservation" if skip_freq_preservation else ""
        os.system(
            f"python3 dbproc.py --dataset-name={dataset_name} --table-name={table_name} --drift={drift}{ref_arg}{skip_freq_arg}"
        )

    return used_validated_cache


def _validate_and_get_failed_tables(
    dataset_name: str,
    table_names: List[str],
    threshold: float = 1.2,
    gen_db: str = "imdb_17v2_gen",
    real_db: str = "imdb_17v2",
    drift_ref: dict = None,  # {table_name: (drift, corr_loss)} for saving validation status
    reference_dataset: str = None,
    variant_id: int = -1,
    dry_run: bool = False,
    validated_cache_tables: set = None,  # Tables already validated - import only, skip SQL
    skip_freq_preservation: bool = False,  # Skip frequency preservation (for non-drift-ref mode)
) -> Tuple[List[str], List[str]]:
    """
    Validate generated tables against real database.

    Returns:
        Tuple[List[str], List[str]]:
            - failed_tables: tables that failed validation (can be retried with auto-tune)
            - import_failed_tables: tables that failed due to import error (don't retry)

    Also saves validation status to cache if drift_ref is provided.
    """
    from db_validation import DatabaseValidator

    # Sort tables to validate: put 'title' last (it's referenced by other tables via FK)
    # This avoids FK constraint issues when using DELETE instead of TRUNCATE CASCADE
    FK_VALIDATION_ORDER = [
        'aka_name', 'person_info',  # Reference: name, info_type
        'aka_title',                 # Reference: kind_type, title
        'movie_companies',           # Reference: title, company_name
        'movie_keyword',             # Reference: title, keyword
        'movie_link',                # Reference: title, link_type
        'title',                     # Referenced BY: movie_*, aka_title (validate LAST)
    ]
    def sort_key(t):
        try:
            return FK_VALIDATION_ORDER.index(t)
        except ValueError:
            return 0  # Unknown tables first
    table_names = sorted(table_names, key=sort_key)

    print(f"\nValidating {len(table_names)} table(s) against database...")
    print(f"  gen_db: {gen_db}")
    print(f"  real_db: {real_db}")
    print(f"  threshold: {threshold:.1%}")
    print(f"  order: {table_names}")
    print()

    validator = DatabaseValidator(
        gen_db=gen_db,
        real_db=real_db,
        port=GLOBAL_CONFIG["pg_port"],
    )

    # Ensure gen_db exists
    if not validator.database_exists(gen_db):
        print(f"Setting up {gen_db} database...")
        if not validator.create_database(gen_db):
            print("Failed to create database, skipping validation")
            return [], []
        if not validator.copy_all_tables(real_db, gen_db):
            print("Failed to copy tables, skipping validation")
            return [], []

    failed_tables = []
    import_failed_tables = []

    if validated_cache_tables is None:
        validated_cache_tables = set()

    for table_name in table_names:
        # Check if this table already passed validation (in cache)
        is_validated = table_name in validated_cache_tables

        if is_validated:
            print(f"\n--- Importing (validated cache): {table_name} ---")
            # Just import the data, skip SQL validation
            if not dry_run:
                regenerate_with_best_cached_params(dataset_name, table_name, skip_freq_preservation=skip_freq_preservation, reference_dataset=reference_dataset, variant_id=variant_id)
                success = validator.import_generated_table(validator.gen_db, dataset_name, table_name,
                                                           reference_dataset=reference_dataset, variant_id=variant_id)
                if success:
                    print(f"  ✓ Imported (already validated)")
                else:
                    print(f"  ✗ Import failed")
                    import_failed_tables.append(table_name)
            else:
                print(f"  [DRY RUN] Would import (already validated)")
        else:
            print(f"\n--- Validating: {table_name} ---")

            # Ensure we have data generated with best cached params
            if not dry_run:
                regenerate_with_best_cached_params(dataset_name, table_name, skip_freq_preservation=skip_freq_preservation, reference_dataset=reference_dataset, variant_id=variant_id)

            result = validator.validate_single_table(dataset_name, table_name, threshold, dry_run=dry_run,
                                                     reference_dataset=reference_dataset, variant_id=variant_id)

            passed = result.passed
            ratio = result.time_ratio

            lower_bound = 1.0 / threshold
            upper_bound = threshold
            if passed:
                print(f"  ✓ PASSED (ratio={ratio:.2f}x in [{lower_bound:.2f}, {upper_bound:.2f}])")
            elif result.import_failed:
                # Import failed - don't retry auto-tune
                print(f"  ✗ IMPORT FAILED - data format error, skipping retry")
                import_failed_tables.append(table_name)
            else:
                print(f"  ✗ FAILED (ratio={ratio:.2f}x outside [{lower_bound:.2f}, {upper_bound:.2f}])")
                failed_tables.append(table_name)

            # Save validation status to cache (only if not import failure)
            if drift_ref and table_name in drift_ref and not result.import_failed:
                target_drift, _ = drift_ref[table_name]
                tuner = AutoTuner(
                    dataset_name=dataset_name,
                    table_name=table_name,
                    reference_dataset=reference_dataset,
                    verbose=False,
                )
                tuner.set_validation_status(target_drift, passed, ratio)

    return failed_tables, import_failed_tables


def _check_cache_status(dataset_name: str, table_name: str, target_drift: float, reference_dataset: str = None) -> dict:
    """
    Check cache status for a table.

    Returns dict with:
        - has_cache: bool - whether cache exists for this table/drift
        - validation_passed: bool - whether validation already passed
        - within_tolerance: bool - whether drift/corr are within tolerance
        - fallback_type: str - if generated with fallback (year_offset, etc.)
        - cache_path: str - path to cache file
        - cache_key: str - key in cache file
    """
    import json

    result = {
        "has_cache": False,
        "validation_passed": False,
        "within_tolerance": False,
        "fallback_type": "",
        "cache_path": None,
        "cache_key": None,
    }

    # Build cache path
    ref_suffix = f"_ref_{reference_dataset}" if reference_dataset else ""
    cache_path = os.path.join("tuning_cache", f"{dataset_name}_{table_name}{ref_suffix}.json")

    if not os.path.exists(cache_path):
        return result

    try:
        with open(cache_path) as f:
            cache_data = json.load(f)

        # Find matching cache key (closest drift value within tolerance)
        cache_key = f"{target_drift:.2f}"
        if cache_key not in cache_data.get("best_params", {}):
            # Try to find close match (within CACHE_KEY_TOLERANCE)
            best_match = None
            best_diff = float('inf')
            for key in cache_data.get("best_params", {}).keys():
                try:
                    cached_drift = float(key)
                    diff = abs(cached_drift - target_drift)
                    if diff < CACHE_KEY_TOLERANCE and diff < best_diff:
                        best_match = key
                        best_diff = diff
                except ValueError:
                    continue
            if best_match:
                cache_key = best_match

        if cache_key not in cache_data.get("best_params", {}):
            return result

        cached_result = cache_data["best_params"][cache_key]

        result["has_cache"] = True
        result["cache_path"] = cache_path
        result["cache_key"] = cache_key
        result["validation_passed"] = cached_result.get("validation_passed", False)
        result["fallback_type"] = cached_result.get("fallback_type", "")

        # Check if drift/corr are within tolerance
        drift_error = cached_result.get("drift_error", 1.0)
        correlation_loss = cached_result.get("correlation_loss", 1.0)
        if drift_error < DRIFT_ERROR_TOLERANCE and correlation_loss < CORRELATION_TOLERANCE:
            result["within_tolerance"] = True

        return result

    except Exception as e:
        print(f"  Warning: Error reading cache for {table_name}: {e}")
        return result


def _generate_with_drift_reference(
    dataset_name: str,
    drift_ref_file: str,
    reference_dataset: Optional[str],
    gpus: Optional[List[int]],
    single_table: Optional[str] = None,
    validate_after: bool = False,
    validate_threshold: float = 1.2,
    max_retries: int = 3,
    dry_run: bool = False,
    ops_mode: str = "retain",  # all, retain, regene-only
    exclude_tables: Optional[List[str]] = None,
    batch_size: int = 524288,
    sample_steps: Optional[int] = None,
    variant_id: int = -1,
):
    """Generate data using drift reference file (per-table target drift and correlation)."""
    # Load drift reference, filtering by dataset_name (src) and reference_dataset (dst)
    drift_ref = load_drift_reference(drift_ref_file, src_dataset=dataset_name, dst_dataset=reference_dataset)
    if drift_ref is None:
        return

    # If single table specified, filter to just that table
    if single_table:
        if single_table not in drift_ref:
            print(f"Error: Table '{single_table}' not found in drift reference file")
            print(f"Available tables: {list(drift_ref.keys())}")
            return
        drift_ref = {single_table: drift_ref[single_table]}

    # Exclude specified tables
    if exclude_tables:
        for t in exclude_tables:
            if t in drift_ref:
                del drift_ref[t]
                print(f"Excluding table: {t}")

    print(f"\n{'='*60}")
    print(f"DRIFT REFERENCE MODE")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Reference file: {drift_ref_file}")
    if single_table:
        print(f"Single table: {single_table}")
    print(f"Reference dataset: {reference_dataset or '(default)'}")
    print(f"Tables to generate: {list(drift_ref.keys())}")
    print(f"GPUs: {gpus if gpus else [0]}")
    print(f"Operation mode: {ops_mode}")
    if validate_after:
        if dry_run:
            print(f"Validation: DRY RUN (no table replacement, threshold={validate_threshold:.1%})")
        else:
            print(f"Validation: enabled (threshold={validate_threshold:.1%})")
    print()

    # Print per-table drift and correlation targets, and check cache status
    print("Target per table:")
    tables_to_skip = set()  # Tables to skip (validation already passed)
    tables_generate_only = set()  # Tables to generate without re-tuning (cache OK, no validation yet)

    for table, (target_drift, target_corr) in drift_ref.items():
        corr_str = f"{target_corr:.4f}" if target_corr is not None else "N/A"
        cache_status = _check_cache_status(dataset_name, table, target_drift, reference_dataset)

        status_str = ""
        if ops_mode == "all":
            # Force retrain: ignore all cache
            status_str = " [RETRAIN: --ops=all]"
        elif ops_mode == "retrain":
            # Force retrain with cached/default params (no tuning)
            if cache_status["has_cache"]:
                status_str = " [RETRAIN: cached params]"
            else:
                status_str = " [RETRAIN: default params]"
        elif cache_status["has_cache"]:
            if cache_status["validation_passed"]:
                if ops_mode == "regene-only":
                    status_str = " [CACHE: validated, REGEN]"
                else:  # retain
                    status_str = " [CACHE: validated, SKIP]"
                    tables_to_skip.add(table)
            elif cache_status["within_tolerance"]:
                status_str = " [CACHE: OK, generate only]"
                tables_generate_only.add(table)
            elif cache_status["fallback_type"]:
                status_str = f" [CACHE: {cache_status['fallback_type']} fallback]"
                tables_generate_only.add(table)
            else:
                status_str = " [CACHE: needs tuning]"
        else:
            status_str = " [NO CACHE: auto-tune]"

        print(f"  {table}: drift={target_drift:.4f}, corr_loss={corr_str}{status_str}")
    print()

    if tables_to_skip:
        print(f"Skipping {len(tables_to_skip)} validated table(s): {list(tables_to_skip)}")
        if ops_mode != "retain":
            print(f"  (--ops={ops_mode} specified, will regenerate anyway)")
            tables_to_skip.clear()
            tables_generate_only.clear()  # Also clear generate-only tables for full retrain
        print()

    # Track tables that need to be generated (exclude skipped ones)
    tables_to_generate = {t: v for t, v in drift_ref.items() if t not in tables_to_skip}

    # If no tables to generate, we're done
    if not tables_to_generate:
        print("All tables already validated. Nothing to generate.")
        print("Use --ops=regene-only to regenerate, or --ops=all to retrain.")
        return

    generated_tables = set()
    validated_cache_tables = set()  # Tables that used validated cache (skip validation)
    retry_counts = {t: 0 for t in drift_ref.keys()}
    force_retrain_tables = set()  # Tables that need forced retrain (validation failed)

    # Load dataset info for n_samples (for GPU allocation)
    # batch_size is passed as parameter (default 524288)
    base_dir = os.path.join("datasets", dataset_name)
    full_dataset_info = load_json(os.path.join(base_dir, "dataset_info.json"))

    # Start timing
    generation_start_time = time.time()

    # Special handling for ops=retrain: use cached/default params, force retrain, no tuning
    if ops_mode == "retrain":
        print(f"\n=== ops=retrain: Force retrain with cached/default params (no tuning) ===")

        if gpus and len(gpus) > 1:
            # Parallel retrain: one table per GPU
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import subprocess

            log_subdir = get_dataset_dir(dataset_name, reference_dataset)
            if variant_id > 0:
                log_subdir += f"-{variant_id}"
            log_dir = os.path.join("gd_logs", log_subdir)
            os.makedirs(log_dir, exist_ok=True)

            def run_retrain(table_name, target_drift, target_corr, gpu_id):
                cmd = f"python3 -u auto_tune.py --dataset-name={dataset_name} --table-name={table_name}"
                cmd += f" --target-drift={target_drift} --device={gpu_id}"
                cmd += " --retrain-only"  # Use cached/default params, force retrain, no tuning
                if reference_dataset:
                    cmd += f" --reference-dataset={reference_dataset}"
                if target_corr is not None:
                    cmd += f" --target-corr-loss={target_corr}"
                if sample_steps:
                    cmd += f" --sample-steps={sample_steps}"
                if variant_id > 0:
                    cmd += f" --variant-id={variant_id}"

                log_file = os.path.join(log_dir, f"{table_name}_retrain.log")
                print(f"[GPU {gpu_id}] Retrain: {table_name} (log: {log_file})")

                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                with open(log_file, "w", buffering=1) as f:
                    proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env)
                    ret = proc.wait()

                success = ret == 0
                status = "Done" if success else "FAIL"
                print(f"[GPU {gpu_id}] {status}: {table_name}")
                return table_name, success

            # Assign tables to GPUs round-robin
            retrain_tasks = list(tables_to_generate.items())
            with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
                futures = {}
                for i, (t, (drift, corr)) in enumerate(retrain_tasks):
                    gpu_id = gpus[i % len(gpus)]
                    futures[executor.submit(run_retrain, t, drift, corr, gpu_id)] = t

                for future in as_completed(futures):
                    table_name, success = future.result()
                    if success:
                        generated_tables.add(table_name)
        else:
            # Single GPU: sequential retrain
            device = gpus[0] if gpus else 0
            for t, (target_drift, target_corr) in list(tables_to_generate.items()):
                print(f"\n[Retrain] {t}: drift={target_drift}")
                tuner = AutoTuner(
                    dataset_name=dataset_name,
                    table_name=t,
                    reference_dataset=reference_dataset,
                    device=device,
                    verbose=True,
                    target_corr_loss=target_corr,
                    sample_steps=sample_steps,
                )
                success, output_path = tuner.generate_with_best_params(target_drift, use_cache=True, retrain=True)
                if success:
                    generated_tables.add(t)
                    print(f"Retrain completed: {output_path}")
                else:
                    print(f"Retrain failed!")

        # Print timing and exit (no validation loop for retrain mode)
        generation_end_time = time.time()
        generation_time = generation_end_time - generation_start_time
        hours, remainder = divmod(generation_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE (ops=retrain)")
        print(f"{'='*60}")
        print(f"Total time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
        print(f"Tables generated: {len(generated_tables)}/{len(drift_ref)}")
        return

    # Special handling for ops=retrain-only: only train, skip generation
    if ops_mode == "retrain-only":
        print(f"\n=== ops=retrain-only: Train only, skip generation ===")

        if gpus and len(gpus) > 1:
            # Parallel training: one table per GPU
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import subprocess

            log_subdir = get_dataset_dir(dataset_name, reference_dataset)
            if variant_id > 0:
                log_subdir += f"-{variant_id}"
            log_dir = os.path.join("gd_logs", log_subdir)
            os.makedirs(log_dir, exist_ok=True)

            def run_train_only(table_name, target_drift, target_corr, gpu_id):
                cmd = f"python3 -u dbproc.py --dataset-name={dataset_name} --table-name={table_name}"
                cmd += f" --drift={target_drift} --device={gpu_id}"
                cmd += " --retrain-diffuser --retrain-controller --train-only"
                if reference_dataset:
                    cmd += f" --reference-dataset={reference_dataset}"
                if variant_id > 0:
                    cmd += f" --variant-id={variant_id}"

                log_file = os.path.join(log_dir, f"{table_name}_train.log")
                print(f"[GPU {gpu_id}] Train: {table_name} (log: {log_file})")

                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                with open(log_file, "w", buffering=1) as f:
                    proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env)
                    ret = proc.wait()

                success = ret == 0
                status = "Done" if success else "FAIL"
                print(f"[GPU {gpu_id}] {status}: {table_name}")
                return table_name, success

            # Assign tables to GPUs round-robin
            train_tasks = list(tables_to_generate.items())
            with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
                futures = {}
                for i, (t, (drift, corr)) in enumerate(train_tasks):
                    gpu_id = gpus[i % len(gpus)]
                    futures[executor.submit(run_train_only, t, drift, corr, gpu_id)] = t

                for future in as_completed(futures):
                    table_name, success = future.result()
                    if success:
                        generated_tables.add(table_name)
        else:
            # Single GPU: sequential training
            device = gpus[0] if gpus else 0
            for t, (target_drift, target_corr) in list(tables_to_generate.items()):
                print(f"\n[Train] {t}: drift={target_drift}")
                tuner = AutoTuner(
                    dataset_name=dataset_name,
                    table_name=t,
                    reference_dataset=reference_dataset,
                    device=device,
                    verbose=True,
                    target_corr_loss=target_corr,
                    sample_steps=sample_steps,
                )
                success, _ = tuner.generate_with_best_params(target_drift, use_cache=True, retrain=True, train_only=True)
                if success:
                    generated_tables.add(t)
                    print(f"Training completed for {t}")
                else:
                    print(f"Training failed for {t}!")

        # Print timing and exit
        generation_end_time = time.time()
        generation_time = generation_end_time - generation_start_time
        hours, remainder = divmod(generation_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE (ops=retrain-only)")
        print(f"{'='*60}")
        print(f"Total time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
        print(f"Tables trained: {len(generated_tables)}/{len(drift_ref)}")
        return

    while tables_to_generate:
        # Separate tables: those needing auto-tune vs those with cached params
        # force_retrain_tables: validation failed, need full retrain (go to auto-tune, not cached)
        tables_need_autotune = {t: v for t, v in tables_to_generate.items()
                                if t not in tables_generate_only or t in force_retrain_tables}
        tables_cached = {t: v for t, v in tables_to_generate.items()
                         if t in tables_generate_only and t not in force_retrain_tables}

        # Phase 1: Run auto-tune for tables without cache (generates full table, no chunking)
        if tables_need_autotune:
            print(f"\n=== Phase 1: Auto-tune ({len(tables_need_autotune)} tables) ===")
            print("These tables need auto-tune and will generate full tables directly.")

            if gpus and len(gpus) > 1:
                # Parallel auto-tune: one table per GPU
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import subprocess

                log_subdir = get_dataset_dir(dataset_name, reference_dataset)
                if variant_id > 0:
                    log_subdir += f"-{variant_id}"
                log_dir = os.path.join("gd_logs", log_subdir)
                os.makedirs(log_dir, exist_ok=True)

                def run_autotune(table_name, target_drift, target_corr, gpu_id):
                    cmd = f"python3 -u auto_tune.py --dataset-name={dataset_name} --table-name={table_name}"
                    cmd += f" --target-drift={target_drift} --device={gpu_id}"
                    if reference_dataset:
                        cmd += f" --reference-dataset={reference_dataset}"
                    if target_corr is not None:
                        cmd += f" --target-corr-loss={target_corr}"
                    if sample_steps:
                        cmd += f" --sample-steps={sample_steps}"
                    if variant_id > 0:
                        cmd += f" --variant-id={variant_id}"
                    # Map ops_mode to auto_tune.py flags, or force no-cache for retrain tables
                    if ops_mode == "all" or table_name in force_retrain_tables:
                        cmd += " --no-cache"
                    elif ops_mode == "regene-only":
                        cmd += " --force-regenerate"

                    log_file = os.path.join(log_dir, f"{table_name}_autotune.log")
                    print(f"[GPU {gpu_id}] Auto-tune: {table_name} (log: {log_file})")

                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    with open(log_file, "w", buffering=1) as f:
                        proc = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env)
                        ret = proc.wait()

                    success = ret == 0
                    status = "Done" if success else "FAIL"
                    print(f"[GPU {gpu_id}] {status}: {table_name}")
                    return table_name, success

                # Assign tables to GPUs round-robin
                autotune_tasks = list(tables_need_autotune.items())
                with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
                    futures = {}
                    for i, (t, (drift, corr)) in enumerate(autotune_tasks):
                        gpu_id = gpus[i % len(gpus)]
                        futures[executor.submit(run_autotune, t, drift, corr, gpu_id)] = t

                    for future in as_completed(futures):
                        table_name, success = future.result()
                        if success:
                            generated_tables.add(table_name)
                            tables_generate_only.add(table_name)  # Now has cache
                        del tables_to_generate[table_name]
            else:
                # Single GPU: sequential auto-tune
                device = gpus[0] if gpus else 0
                for t, (target_drift, target_corr) in list(tables_need_autotune.items()):
                    # Map ops_mode to internal flags, or force no-cache for retrain tables
                    force_regenerate = (ops_mode == "regene-only")
                    no_cache = (ops_mode == "all") or (t in force_retrain_tables)
                    _generate_with_auto_tune(
                        dataset_name, t, target_drift, reference_dataset, device,
                        force_regenerate=force_regenerate, no_cache=no_cache,
                        target_corr_loss=target_corr,
                        sample_steps=sample_steps,
                    )
                    generated_tables.add(t)
                    del tables_to_generate[t]

            print(f"=== Phase 1 complete ===\n")

        # Phase 2: Chunked parallel generation for tables with cached params
        if not tables_to_generate:
            break  # All done

        # Update tables_cached after auto-tune phase
        tables_cached = {t: v for t, v in tables_to_generate.items()}

        if not tables_cached:
            break

        if gpus and len(gpus) > 1:
            # Multi-GPU parallel execution with smart allocation based on n_samples
            print(f"=== Phase 2: Chunked generation ({len(tables_cached)} tables) ===")
            print(f"Running parallel generation on GPUs: {gpus}")
            print()

            # Calculate batch count for each table and sort by size (largest first)
            # Use reference dataset row count if available, else fallback to dataset_info.json
            ref_dir = os.path.join("datasets", reference_dataset) if reference_dataset else base_dir
            table_batch_info = []
            for t in tables_to_generate.keys():
                # Try to get row count from reference dataset CSV
                ref_csv = os.path.join(ref_dir, f"{t}.csv")
                if os.path.exists(ref_csv):
                    # Count lines (subtract 1 for header)
                    with open(ref_csv, 'r') as f:
                        n_samples = sum(1 for _ in f) - 1
                else:
                    table_info = full_dataset_info.get(t, {})
                    n_samples = table_info.get("n_samples", 100000) if table_info else 100000
                n_batches = (n_samples + batch_size - 1) // batch_size
                target_drift, target_corr = tables_to_generate[t]
                table_batch_info.append((t, n_samples, n_batches, target_drift, target_corr))

            # Sort by n_samples descending (process large tables first for better balancing)
            table_batch_info.sort(key=lambda x: -x[1])

            num_tables = len(table_batch_info)
            num_gpus = len(gpus)
            total_batches = sum(x[2] for x in table_batch_info)
            ideal_per_gpu = total_batches / num_gpus

            print(f"Scheduling: {total_batches} batches, {num_tables} tables, {num_gpus} GPUs (batch_size={batch_size})")
            print(f"  Ideal load per GPU: {ideal_per_gpu:.1f} batches")

            # Check if largest table benefits from multi-GPU splitting
            largest = table_batch_info[0]
            largest_t, largest_samples, largest_batches, largest_drift, largest_corr = largest
            remaining_tables = table_batch_info[1:]
            remaining_batches = sum(x[2] for x in remaining_tables)

            # Use multi-GPU for largest table if:
            # 1. It has >= 4 batches
            # 2. It's significantly larger than ideal (> 1.2x)
            # 3. Remaining tables can fit on remaining GPUs
            use_multi_gpu = False
            multi_gpu_count = 0

            if largest_batches >= 4 and largest_batches > ideal_per_gpu * 1.2:
                # How many GPUs for the largest table?
                # We want: largest_batches / n <= ideal_per_gpu
                # So: n >= largest_batches / ideal_per_gpu
                min_gpus_needed = max(2, int(largest_batches / ideal_per_gpu + 0.5))

                # But we need GPUs for other large tables too
                # Count how many tables have >= 4 batches (need their own GPU)
                large_tables = [t for t in remaining_tables if t[2] >= 4]
                gpus_for_others = len(large_tables)

                available_for_multi = num_gpus - gpus_for_others
                if available_for_multi >= 2:
                    multi_gpu_count = min(min_gpus_needed, available_for_multi, largest_batches)
                    use_multi_gpu = True

            # Build work pool: each batch is an independent task
            # GPUs dynamically pull from queue, 2 concurrent tasks per GPU
            max_concurrent_per_gpu = 2

            # Split each table into batches (each batch = batch_size samples)
            # batch_task = (table, batch_idx, sample_start, sample_count, drift, corr)
            # Phase 2: all batches use cached params, all batches are independent
            table_batch_counts = {}
            table_samples = {}  # table -> total samples

            # Group batches by table for interleaving
            table_batches = {}
            for t, n_samples, n_batches, target_drift, target_corr in table_batch_info:
                table_batch_counts[t] = n_batches
                table_samples[t] = n_samples
                table_batches[t] = []
                for batch_idx in range(n_batches):
                    sample_start = batch_idx * batch_size
                    sample_count = min(batch_size, n_samples - sample_start)
                    batch = (t, batch_idx, sample_start, sample_count, target_drift, target_corr)
                    table_batches[t].append(batch)

            # Interleave batches from different tables (reduces same-table conflicts on same GPU)
            all_batches = []
            while any(table_batches.values()):
                for t in list(table_batches.keys()):
                    if table_batches[t]:
                        all_batches.append(table_batches[t].pop(0))
                    if not table_batches[t]:
                        del table_batches[t]

            print(f"  Work pool: {len(all_batches)} batches from {len(table_batch_info)} tables")
            print(f"  Workers: {num_gpus} GPUs x {max_concurrent_per_gpu} slots = {num_gpus * max_concurrent_per_gpu} parallel")
            for t, count in table_batch_counts.items():
                print(f"    {t}: {count} batches")

            # Estimate time: total_batches / num_slots (parallel execution)
            num_slots = num_gpus * max_concurrent_per_gpu
            estimated_time = (total_batches + num_slots - 1) // num_slots  # ceil
            print(f"  Estimated time: ~{estimated_time} batch units (~{estimated_time * 3.5:.0f} min)")
            print()

            # Launch processes using work pool pattern
            log_subdir = get_dataset_dir(dataset_name, reference_dataset)
            if variant_id > 0:
                log_subdir += f"-{variant_id}"
            log_dir = os.path.join("gd_logs", log_subdir)
            os.makedirs(log_dir, exist_ok=True)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from queue import Queue
            import threading

            # Work queue with all batches
            work_queue = Queue()
            for batch in all_batches:
                work_queue.put(batch)

            # Results storage
            batch_results = []
            results_lock = threading.Lock()

            # Track which tables are currently running on each GPU (to avoid OOM)
            gpu_running_tables = {g: set() for g in gpus}
            gpu_tables_lock = threading.Lock()

            def run_batch(table_name, batch_idx, sample_start, sample_count, gpu_id, target_drift, target_corr):
                """Run a single batch on specified GPU."""
                # Phase 2: All tables have cached params, use dbproc.py directly
                cached_params = get_cached_params_for_table(dataset_name, table_name, target_drift, reference_dataset)
                cmd = f"python3 -u dbproc.py --dataset-name={dataset_name} --table-name={table_name}"
                cmd += f" --drift={target_drift} --device={gpu_id}"
                cmd += f" --sample-start={sample_start} --sample-count={sample_count}"
                if cached_params:
                    cmd += f" {cached_params.to_cmd_args()}"
                cmd += " --reuse"  # Use existing models (trained in Phase 1)
                if reference_dataset:
                    cmd += f" --reference-dataset={reference_dataset}"
                if sample_steps:
                    cmd += f" --sample-steps={sample_steps}"
                if variant_id > 0:
                    cmd += f" --variant-id={variant_id}"

                batch_str = f"{table_name}[{batch_idx}]"
                print(f"[GPU {gpu_id}] Start: {batch_str}")

                log_file_path = os.path.join(log_dir, f"{table_name}_b{batch_idx}.log")
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"

                with open(log_file_path, "w", buffering=1) as log_file:
                    proc = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT, env=env)
                    ret = proc.wait()
                    success = (ret == 0)
                    status = "Done" if success else "FAIL"
                    print(f"[GPU {gpu_id}] {status}: {batch_str}")
                    return (table_name, batch_idx, sample_start, sample_count, gpu_id, success, log_file_path)

            def gpu_worker(gpu_id):
                """Worker that pulls batches from queue and runs them on this GPU."""
                skipped = []  # Batches we couldn't run (table conflict)

                while True:
                    batch = None

                    # First try skipped batches
                    for i, b in enumerate(skipped):
                        t = b[0]
                        with gpu_tables_lock:
                            if t not in gpu_running_tables[gpu_id]:
                                batch = skipped.pop(i)
                                break

                    # Then try queue
                    if batch is None:
                        try:
                            batch = work_queue.get_nowait()
                        except:
                            if skipped:
                                # Still have skipped batches, wait and retry
                                time.sleep(0.5)
                                continue
                            break  # Queue empty and no skipped

                    t, batch_idx, sample_start, sample_count, target_drift, target_corr = batch

                    # Check if this table is already running on this GPU (would cause OOM)
                    with gpu_tables_lock:
                        if t in gpu_running_tables[gpu_id]:
                            skipped.append(batch)
                            continue
                        gpu_running_tables[gpu_id].add(t)

                    # Phase 2: No training needed, all batches can run independently
                    try:
                        result = run_batch(t, batch_idx, sample_start, sample_count, gpu_id, target_drift, target_corr)
                        with results_lock:
                            batch_results.append(result)
                    finally:
                        with gpu_tables_lock:
                            gpu_running_tables[gpu_id].discard(t)

            # Start workers: 2 per GPU, with retry for failed batches
            max_retries = 2
            for retry_round in range(max_retries + 1):
                if retry_round == 0:
                    print(f"=== Starting {num_gpus * max_concurrent_per_gpu} workers ===")
                else:
                    # Collect failed batches and retry
                    # batch_results: (table, batch_idx, sample_start, sample_count, gpu, success, log)
                    failed_batches = []
                    for t, idx, start, count, gpu, success, log in batch_results:
                        if not success:
                            drift, corr = drift_ref.get(t, (0.3, None))
                            failed_batches.append((t, idx, start, count, drift, corr))

                    if not failed_batches:
                        break
                    print(f"\n=== Retry round {retry_round}: {len(failed_batches)} failed batches ===")

                    # Reset for retry
                    work_queue = Queue()
                    for batch in failed_batches:
                        work_queue.put(batch)
                    batch_results = [r for r in batch_results if r[5]]  # Keep only successful

                workers = []
                with ThreadPoolExecutor(max_workers=num_gpus * max_concurrent_per_gpu) as executor:
                    for gpu_id in gpus:
                        for slot in range(max_concurrent_per_gpu):
                            workers.append(executor.submit(gpu_worker, gpu_id))

                    for w in workers:
                        w.result()

            # Count final results
            failed_count = sum(1 for r in batch_results if not r[5])
            print(f"\n=== All batches completed: {len(batch_results) - failed_count} succeeded, {failed_count} failed ===")

            # Post-process: merge if needed, then apply freq preservation if needed
            print("\n=== Post-processing ===")
            for t, n_batches in table_batch_counts.items():
                save_dir = get_expdir_path(dataset_name, t, reference_dataset, variant_id)
                target_drift = drift_ref.get(t, (0.3, None))[0]

                # Collect .npy files
                npy_files = []
                for batch_idx in range(n_batches):
                    sample_start = batch_idx * batch_size
                    sample_end = min(sample_start + batch_size, table_samples[t])
                    npy_file = os.path.join(save_dir, f"{t}.normalized.chunk_{sample_start}_{sample_end}.npy")
                    if os.path.exists(npy_file):
                        npy_files.append(npy_file)

                if len(npy_files) != n_batches:
                    print(f"  {t}: ERROR - missing .npy files ({len(npy_files)}/{n_batches})")
                    continue

                # Step 1: Merge if multiple chunks
                if n_batches > 1:
                    print(f"  {t}: merging {n_batches} chunks...")

                # Step 2: Apply freq preservation and save
                final_df = _merge_npy_with_freq_preservation(
                    dataset_name, t, npy_files, reference_dataset, target_drift
                )
                if final_df is not None:
                    merged_file = os.path.join(save_dir, f"{t}.drifted.csv")
                    final_df.to_csv(merged_file, index=False, doublequote=False, escapechar="\\")
                    print(f"  {t}: -> {len(final_df)} rows")
                    # Cleanup
                    for npy_f in npy_files:
                        os.remove(npy_f)
                    for batch_idx in range(n_batches):
                        sample_start = batch_idx * batch_size
                        sample_end = min(sample_start + batch_size, table_samples[t])
                        csv_f = os.path.join(save_dir, f"{t}.drifted.chunk_{sample_start}_{sample_end}.csv")
                        if os.path.exists(csv_f):
                            os.remove(csv_f)
                else:
                    print(f"  {t}: ERROR - post-processing failed")

            # Print total time before validation
            generation_end_time = time.time()
            gen_time = generation_end_time - generation_start_time
            gen_hours, gen_remainder = divmod(gen_time, 3600)
            gen_minutes, gen_seconds = divmod(gen_remainder, 60)

            print(f"\n{'='*60}")
            print(f"All tables processed")
            if gen_hours > 0:
                print(f"Total generation time: {int(gen_hours)}h {int(gen_minutes)}m {gen_seconds:.1f}s")
            elif gen_minutes > 0:
                print(f"Total generation time: {int(gen_minutes)}m {gen_seconds:.1f}s")
            else:
                print(f"Total generation time: {gen_seconds:.1f}s")
            print(f"{'='*60}")

            # Validate generated data against original and target
            _validate_generation_results(
                dataset_name,
                list(table_batch_counts.keys()),
                drift_ref,
            )

            # Collect results by table
            table_success = {}
            for t, batch_idx, sample_start, sample_count, gpu_id, success, log_path in batch_results:
                if t not in table_success:
                    table_success[t] = True
                table_success[t] = table_success[t] and success

            all_results = [(t, 0, 0, 0, success, "") for t, success in table_success.items()]

            # Collect results
            gen_failed = [t for t, _, _, _, success, _ in all_results if not success]
            for t, gpu, target_drift, target_corr, success, log_path in all_results:
                if success:
                    generated_tables.add(t)

            print(f"\nGeneration complete: {len(generated_tables)}/{len(tables_to_generate)} succeeded")
            if gen_failed:
                print(f"Failed: {gen_failed}")

        else:
            # Sequential execution
            device = gpus[0] if gpus else 0

            for table_name, (target_drift, target_corr) in tables_to_generate.items():
                corr_str = f", corr_loss={target_corr:.4f}" if target_corr is not None else ""
                if table_name in force_retrain_tables:
                    mode_str = f" [RETRY {retry_counts[table_name]}]"
                elif table_name in tables_generate_only:
                    mode_str = " [GEN ONLY]"
                else:
                    mode_str = ""
                print(f"\n{'='*60}")
                print(f"Generating: {table_name} (drift={target_drift:.4f}{corr_str}) on GPU {device}{mode_str}")
                print(f"{'='*60}")

                # Force no-cache for ops=all or validation-failed tables
                no_cache = (ops_mode == "all") or (table_name in force_retrain_tables)

                # Pass target_corr_loss from drift_ref.csv (base vs ref correlation loss)
                # Auto-tune will compare measured (base vs generated) with target (base vs ref)
                used_validated_cache = _generate_with_auto_tune(
                    dataset_name, table_name, target_drift, reference_dataset, device,
                    target_corr_loss=target_corr, no_cache=no_cache, require_validation=validate_after,
                    sample_steps=sample_steps,
                )
                generated_tables.add(table_name)
                if used_validated_cache:
                    validated_cache_tables.add(table_name)

        # Clear tables_to_generate for now (will re-add if validation fails)
        tables_to_generate = {}

        # Validation phase
        # Process all generated tables: validated ones just import, others run full validation
        if validate_after and generated_tables:
            print(f"\n{'#'*60}")
            print(f"VALIDATION PHASE")
            print(f"{'#'*60}")
            tables_needing_validation = generated_tables - validated_cache_tables
            if validated_cache_tables:
                print(f"Already validated (import only): {list(validated_cache_tables)}")
            if tables_needing_validation:
                print(f"Need validation: {list(tables_needing_validation)}")

            # Pass all generated tables, with validated_cache_tables for import-only handling
            failed_tables, import_failed_tables = _validate_and_get_failed_tables(
                dataset_name, list(generated_tables), validate_threshold,
                drift_ref=drift_ref, reference_dataset=reference_dataset,
                variant_id=variant_id,
                dry_run=dry_run,
                validated_cache_tables=validated_cache_tables,
                skip_freq_preservation=False,  # Use synthetic distribution for freq preservation
            )

            # Handle import failures (don't retry - data format error)
            if import_failed_tables:
                print(f"\n⚠ Import failed (data format error): {import_failed_tables}")
                print(f"  These tables will NOT be retried - please check CSV format")
                # Mark as max retries reached so they won't be retried
                for t in import_failed_tables:
                    retry_counts[t] = max_retries + 1

            if failed_tables:
                print(f"\nFailed validation: {failed_tables}")

                # Check which tables can be retried
                for t in failed_tables:
                    if retry_counts[t] < max_retries:
                        retry_counts[t] += 1
                        tables_to_generate[t] = drift_ref[t]
                        force_retrain_tables.add(t)  # Mark for forced retrain
                        print(f"  {t}: will retry (attempt {retry_counts[t]}/{max_retries}) with forced retrain")
                    else:
                        print(f"  {t}: max retries reached, skipping")

                if tables_to_generate:
                    print(f"\nRetrying {len(tables_to_generate)} table(s) with --no-cache...")
                    generated_tables -= set(tables_to_generate.keys())
            elif not import_failed_tables:
                print(f"\n✓ All tables processed successfully!")

    # Calculate total time
    generation_total_time = time.time() - generation_start_time
    hours, remainder = divmod(generation_total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'='*60}")
    print(f"All tables processed using drift reference file")
    if validate_after:
        passed = len([t for t in drift_ref if retry_counts.get(t, 0) <= max_retries])
        print(f"Validation: {passed}/{len(drift_ref)} passed")
    if hours > 0:
        print(f"Total time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    elif minutes > 0:
        print(f"Total time: {int(minutes)}m {seconds:.1f}s")
    else:
        print(f"Total time: {seconds:.1f}s")
    print(f"{'='*60}")


def handle_delete_data(tokens: List[str]):
    tokens = tokens[1:]

    if len(tokens) < 1:
        print("Error: Not enough arguments")
        return

    dataset_name = tokens[0]
    table_name = tokens[1] if len(tokens) > 1 else ""

    exp_dir = "expdir"
    src_dir = os.path.join(exp_dir, dataset_name)
    dest_base_dir = os.path.join(".trash", exp_dir)

    if table_name:
        print_args(
            dataset_name=dataset_name,
            table_name=table_name,
        )

        src = os.path.join(src_dir, table_name)
        if not os.path.exists(src):
            print(
                f"Error: No data generator model found for table {table_name} in dataset {dataset_name}"
            )
            return

        if button_dialog(
            title="Delete data generator model",
            text=f"Do you want to move data generator model for table {table_name} in dataset {dataset_name} to the trash folder (.trash)?",
            buttons=[
                ("No", False),
                ("Yes", True),
            ],
        ).run():
            dest_dir = os.path.join(dest_base_dir, dataset_name)
            os.makedirs(dest_dir, exist_ok=True)

            dst = os.path.join(dest_dir, f"{table_name}_{time.time()}")
            shutil.move(src, dst)
            print(f"Moved {src} to {dst}")
    else:
        print_args(dataset_name=dataset_name)

        if button_dialog(
            title="Delete data generator model",
            text=f"Do you want to move data generator model for all tables in dataset {dataset_name} to the trash folder (.trash)?",
            buttons=[
                ("No", False),
                ("Yes", True),
            ],
        ).run():
            src = src_dir
            dst = os.path.join(dest_base_dir, f"{dataset_name}_{time.time()}")
            shutil.move(src, dst)
            print(f"Moved {src} to {dst}")


def handle_tqo(tokens: List[str]):
    """Handle training learned query optimizer command"""
    tokens = tokens[1:]
    
    if len(tokens) < 1:
        print("Error: Please specify LQO_NAME")
        print("Usage: tqo [LQO_NAME]")
        print("Available LQO: bao, balsa, hybridqo, lero")
        print("Current global settings:")
        print(f"  Dataset: {GLOBAL_CONFIG['dataset']}")
        print(f"  Drift: {GLOBAL_CONFIG['drift']}")
        print("Use 'set' command to change global settings")
        print("Example: set drift 0.5")
        return
    
    lqo_name = tokens[0].lower()

    # Use global configuration
    dataset = GLOBAL_CONFIG["dataset"]
    drift = GLOBAL_CONFIG["drift"]
    query_set = GLOBAL_CONFIG.get("query_set", None)

    print("Training learned query optimizer...")
    print_args(
        lqo_name=lqo_name,
        dataset=dataset,
        drift=drift,
        query_set=query_set if query_set else "(using dataset default)"
    )
    
    if lqo_name == "bao":
        print("Training Bao learned query optimizer...")

        # Check if bao directory exists
        bao_dir = os.path.join("benchmarks", "lqos", "bao")
        if not os.path.exists(bao_dir):
            print(f"Error: Bao directory not found at {bao_dir}")
            return

        # Check if train_bao.py script exists
        train_script = os.path.join(bao_dir, "train_bao.py")
        if not os.path.exists(train_script):
            print(f"Error: Bao training script not found at {train_script}")
            return

        # Determine query directory based on query_set or dataset
        if query_set:
            # Use specified query set
            query_dir = os.path.join("queries", query_set, "train")
            db_name = dataset if dataset else "imdb"
        elif dataset:
            # Use default dataset queries
            query_dir = os.path.join("queries", dataset, "train")
            db_name = dataset
        else:
            print("Error: Please set dataset first using 'set dataset [DATASET_NAME]'")
            return

        # Check if query directory exists
        if not os.path.exists(query_dir):
            print(f"Error: Query directory not found: {query_dir}")
            return

        # Generate output file name in bao_logs_all directory
        import time as time_module
        timestamp = time_module.strftime("%Y%m%d_%H%M%S")
        query_set_name = query_set if query_set else dataset
        log_dir = "bao_logs_all"
        os.makedirs(log_dir, exist_ok=True)
        output_file = os.path.join(log_dir, f"train_{query_set_name}_{timestamp}.txt")

        # Build command with parameters for train_bao.py
        cmd = f"cd {bao_dir} && python3 train_bao.py"
        cmd += f" --query-dir ../../../{query_dir}"
        cmd += f" --database-name {db_name}"
        cmd += f" --output-file ../../../{output_file}"
        cmd += f" --db-port {GLOBAL_CONFIG['pg_port']}"

        print(f"Using query-based training with {query_dir}")
        print(f"Database: {db_name}")
        print(f"PostgreSQL Port: {GLOBAL_CONFIG['pg_port']}")
        print(f"Output file: {output_file}")

        # Run the training script
        print("Starting Bao training pipeline...")
        result = os.system(cmd)

        if result == 0:
            print("[SUCCESS] Bao training completed successfully!")
            print(f"Results saved to: {output_file}")
        else:
            print(f"[FAILED] Bao training failed with exit code {result}")
            sys.exit(1)
    
    elif lqo_name == "balsa":
        print("Training Balsa learned query optimizer...")
        
        # Check if balsa directory exists
        balsa_dir = os.path.join("benchmarks", "lqos", "balsa")
        if not os.path.exists(balsa_dir):
            print(f"Error: Balsa directory not found at {balsa_dir}")
            return
        
        # Check if train_balsa.py script exists
        train_script = os.path.join(balsa_dir, "train_balsa.py")
        if not os.path.exists(train_script):
            print(f"Error: Balsa training script not found at {train_script}")
            return
        
        # Run the training script
        print("Starting Balsa training pipeline...")
        result = os.system(f"cd {balsa_dir} && python3 train_balsa.py")
        
        if result == 0:
            print("[SUCCESS] Balsa training completed successfully!")
        else:
            print(f"[FAILED] Balsa training failed with exit code {result}")
    
    elif lqo_name == "hybridqo":
        print("Training HybridQO learned query optimizer...")
        
        # Check if hybrid_qo directory exists
        hybridqo_dir = os.path.join("benchmarks", "lqos", "hybrid_qo")
        if not os.path.exists(hybridqo_dir):
            print(f"Error: HybridQO directory not found at {hybridqo_dir}")
            return
        
        # Check if train_hybridqo.py script exists
        train_script = os.path.join(hybridqo_dir, "train_hybridqo.py")
        if not os.path.exists(train_script):
            print(f"Error: HybridQO training script not found at {train_script}")
            return
        
        # Run the training script
        print("Starting HybridQO training pipeline...")
        result = os.system(f"cd {hybridqo_dir} && python3 train_hybridqo.py")
        
        if result == 0:
            print("[SUCCESS] HybridQO training completed successfully!")
        else:
            print(f"[FAILED] HybridQO training failed with exit code {result}")
    
    elif lqo_name == "lero":
        print("Training Lero learned query optimizer...")
        
        # Check if Lero directory exists
        lero_dir = os.path.join("benchmarks", "lqos", "Lero")
        if not os.path.exists(lero_dir):
            print(f"Error: Lero directory not found at {lero_dir}")
            return
        
        # Check if train_lero.py script exists
        train_script = os.path.join(lero_dir, "train_lero.py")
        if not os.path.exists(train_script):
            print(f"Error: Lero training script not found at {train_script}")
            return
        
        # Run the training script
        print("Starting Lero training pipeline...")
        result = os.system(f"cd {lero_dir} && python3 train_lero.py")
        
        if result == 0:
            print("[SUCCESS] Lero training completed successfully!")
        else:
            print(f"[FAILED] Lero training failed with exit code {result}")
            
    else:
        print(f"Error: Unknown LQO '{lqo_name}'")
        print("Available LQO: bao, balsa, hybridqo, lero")
        return


def handle_iqo(tokens: List[str]):
    """Handle inference learned query optimizer command"""
    tokens = tokens[1:]

    if len(tokens) < 1:
        print("Error: Please specify LQO_NAME")
        print("Usage: iqo [LQO_NAME] [MODE]")
        print("Available LQO: bao, balsa, hybridqo, lero")
        print("For bao: iqo bao [bao|pg]  (default: bao)")
        print("  - bao: Test with Bao optimizer")
        print("  - pg:  Test with PostgreSQL optimizer")
        print("Current global settings:")
        print(f"  Dataset: {GLOBAL_CONFIG['dataset']}")
        print(f"  Drift: {GLOBAL_CONFIG['drift']}")
        print("Use 'set' command to change global settings")
        print("Example: set drift 0.5")
        return

    lqo_name = tokens[0].lower()

    # Check for additional mode parameter (for bao)
    test_mode = "bao"  # default to bao
    if len(tokens) > 1:
        test_mode = tokens[1].lower()
    
    # Use global configuration
    dataset = GLOBAL_CONFIG["dataset"]
    drift = GLOBAL_CONFIG["drift"]
    query_set = GLOBAL_CONFIG.get("query_set", None)

    print("Running learned query optimizer inference...")
    print_args(
        lqo_name=lqo_name,
        dataset=dataset,
        drift=drift,
        query_set=query_set if query_set else "(using dataset default)"
    )
    
    if lqo_name == "bao":
        # Validate test mode
        if test_mode not in ["bao", "pg"]:
            print(f"Error: Invalid test mode '{test_mode}'")
            print("Usage: iqo bao [bao|pg]")
            print("  - bao: Test with Bao optimizer (default)")
            print("  - pg:  Test with PostgreSQL optimizer")
            return

        mode_name = "Bao" if test_mode == "bao" else "PostgreSQL"
        print(f"Testing with {mode_name} optimizer...")
        print_args(lqo_name=lqo_name, test_mode=test_mode)

        # Check if bao directory exists
        bao_dir = os.path.join("benchmarks", "lqos", "bao")
        if not os.path.exists(bao_dir):
            print(f"Error: Bao directory not found at {bao_dir}")
            return

        # Check if test script exists
        test_script = os.path.join(bao_dir, "test_bao.py")
        if not os.path.exists(test_script):
            print(f"Error: Bao test script not found at {test_script}")
            return

        # Determine query directory based on query_set or dataset
        if query_set:
            # Use specified query set
            query_dir = os.path.join("queries", query_set, "test")
            db_name = dataset if dataset else "imdb"
        elif dataset:
            # Use default dataset queries
            query_dir = os.path.join("queries", dataset, "test")
            db_name = dataset
        else:
            print("Error: Please set dataset first using 'set dataset [DATASET_NAME]'")
            return

        # Check if query directory exists
        if not os.path.exists(query_dir):
            print(f"Error: Query directory not found: {query_dir}")
            return

        # Generate output file name in bao_logs_all directory
        import time as time_module
        timestamp = time_module.strftime("%Y%m%d_%H%M%S")
        query_set_name = query_set if query_set else dataset
        log_dir = "bao_logs_all"
        os.makedirs(log_dir, exist_ok=True)
        output_file = os.path.join(log_dir, f"test_{test_mode}_{query_set_name}_{timestamp}.txt")

        # Build command with parameters for test_bao.py
        cmd = f"cd {bao_dir} && python3 test_bao.py"
        cmd += f" --query-dir ../../../{query_dir}"
        cmd += f" --database-name {db_name}"
        cmd += f" --output-file ../../../{output_file}"
        cmd += f" --db-port {GLOBAL_CONFIG['pg_port']}"

        if test_mode == "bao":
            cmd += " --use-bao"
            print(f"Using Bao optimizer for testing with {query_dir}")
            print(f"Database: {db_name}")
            print(f"PostgreSQL Port: {GLOBAL_CONFIG['pg_port']}")
            print(f"Output file: {output_file}")
        else:  # pg
            cmd += " --use-postgres"
            print(f"Using PostgreSQL optimizer for testing with {query_dir}")
            print(f"Database: {db_name}")
            print(f"PostgreSQL Port: {GLOBAL_CONFIG['pg_port']}")
            print(f"Output file: {output_file}")

        # Run the test script
        print(f"Starting {mode_name} testing pipeline...")
        result = os.system(cmd)

        if result == 0:
            print(f"[SUCCESS] {mode_name} testing completed successfully!")
            print(f"Results saved to: {output_file}")
        else:
            print(f"[FAILED] {mode_name} testing failed with exit code {result}")
            sys.exit(1)
    
    elif lqo_name == "balsa":
        print("Running Balsa learned query optimizer inference...")
        print_args(lqo_name=lqo_name)
        
        # Check if balsa directory exists
        balsa_dir = os.path.join("benchmarks", "lqos", "balsa")
        if not os.path.exists(balsa_dir):
            print(f"Error: Balsa directory not found at {balsa_dir}")
            return
        
        # Check if inference script exists
        inference_script = os.path.join(balsa_dir, "inference_balsa.py")
        if not os.path.exists(inference_script):
            print(f"Error: Balsa inference script not found at {inference_script}")
            return
        
        # Run the inference script
        print("Starting Balsa inference...")
        result = os.system(f"cd {balsa_dir} && python3 inference_balsa.py")
        
        if result == 0:
            print("[SUCCESS] Balsa inference completed successfully!")
        else:
            print(f"[FAILED] Balsa inference failed with exit code {result}")
    
    elif lqo_name == "hybridqo":
        print("Running HybridQO learned query optimizer inference...")
        print_args(lqo_name=lqo_name)
        
        # Check if hybrid_qo directory exists
        hybridqo_dir = os.path.join("benchmarks", "lqos", "hybrid_qo")
        if not os.path.exists(hybridqo_dir):
            print(f"Error: HybridQO directory not found at {hybridqo_dir}")
            return
        
        # Check if inference script exists
        inference_script = os.path.join(hybridqo_dir, "inference_hybridqo.py")
        if not os.path.exists(inference_script):
            print(f"Error: HybridQO inference script not found at {inference_script}")
            return
        
        # Run the inference script
        print("Starting HybridQO inference...")
        result = os.system(f"cd {hybridqo_dir} && python3 inference_hybridqo.py")
        
        if result == 0:
            print("[SUCCESS] HybridQO inference completed successfully!")
        else:
            print(f"[FAILED] HybridQO inference failed with exit code {result}")
    
    elif lqo_name == "lero":
        print("Running Lero learned query optimizer inference...")
        print_args(lqo_name=lqo_name)
        
        # Check if Lero directory exists
        lero_dir = os.path.join("benchmarks", "lqos", "Lero")
        if not os.path.exists(lero_dir):
            print(f"Error: Lero directory not found at {lero_dir}")
            return
        
        # Check if inference script exists
        inference_script = os.path.join(lero_dir, "inference_lero.py")
        if not os.path.exists(inference_script):
            print(f"Error: Lero inference script not found at {inference_script}")
            return
        
        # Run the inference script
        print("Starting Lero inference...")
        result = os.system(f"cd {lero_dir} && python3 inference_lero.py")
        
        if result == 0:
            print("[SUCCESS] Lero inference completed successfully!")
        else:
            print(f"[FAILED] Lero inference failed with exit code {result}")
            
    else:
        print(f"Error: Unknown LQO '{lqo_name}'")
        print("Available LQO: bao, balsa, hybridqo, lero")
        return


def handle_set(tokens: List[str]):
    """Handle set command for global configuration"""
    tokens = tokens[1:]
    
    if len(tokens) == 0:
        # Show current configuration
        print("Current global configuration:")
        print("-" * 30)
        for key, value in GLOBAL_CONFIG.items():
            print(f"{key}: {value}")
        return
    
    if len(tokens) < 2:
        print("Error: Please specify KEY and VALUE")
        print("Usage: set [KEY] [VALUE]")
        print("Available keys:")
        print("  dataset: imdb, books, fb, osm, wiki")
        print("  drift: 0.0-1.0")
        print("  query_set: any query set name (or 'none' to use default)")
        print("Examples:")
        print("  set dataset books")
        print("  set drift 0.5")
        print("  set query_set join-order-benchmark")
        print("  set query_set none  # Use default queries/{dataset}/train|test")
        return
    
    key = tokens[0].lower()
    value = tokens[1]
    
    # Validate and set configuration
    if key == "dataset":
        valid_datasets = ["imdb", "books", "fb", "osm", "wiki"]
        if value.lower() not in valid_datasets:
            print(f"Error: Invalid dataset '{value}'")
            print(f"Available datasets: {', '.join(valid_datasets)}")
            return
        GLOBAL_CONFIG[key] = value.lower()
        print(f"[SUCCESS] Set {key} = {value.lower()}")
        
    elif key == "drift":
        try:
            drift_float = float(value)
            if not 0.0 <= drift_float <= 1.0:
                print("Error: Drift must be between 0.0 and 1.0")
                return
            GLOBAL_CONFIG[key] = drift_float
            print(f"[SUCCESS] Set {key} = {drift_float}")
        except ValueError:
            print("Error: Drift must be a valid number")
            return

    elif key == "query_set":
        if value.lower() == "none":
            GLOBAL_CONFIG[key] = None
            print(f"[SUCCESS] Set {key} = None (using default queries/{{dataset}}/train|test)")
        else:
            GLOBAL_CONFIG[key] = value
            print(f"[SUCCESS] Set {key} = {value}")
            print(f"Query paths will be: queries/{value}/train and queries/{value}/test")

    else:
        print(f"Error: Unknown configuration key '{key}'")
        print("Available keys: dataset, drift, query_set")
        return


def handle_idx(tokens: List[str]):
    """Handle learned index testing command"""
    tokens = tokens[1:]
    
    if len(tokens) < 1:
        print("Error: Please specify INDEX_NAME")
        print("Usage: idx [INDEX_NAME]")
        print("Available INDEX: alex, art, btree, pgm, xindex, finedex")
        print("Current global settings:")
        print(f"  Dataset: {GLOBAL_CONFIG['dataset']}")
        print(f"  Drift: {GLOBAL_CONFIG['drift']}")
        print("Use 'set' command to change global settings")
        print("Example: set drift 0.5")
        return
    
    index_name = tokens[0].lower()
    
    # Use global configuration
    dataset = GLOBAL_CONFIG["dataset"]
    drift = GLOBAL_CONFIG["drift"]
    
    # Set default values for LIDX-specific parameters
    size = "4M"  # Default size for LIDX
    operations = 1000000  # Default operations
    threads = 4  # Default threads
    
    print("Testing learned index...")
    print_args(
        index_name=index_name,
        drift=drift,
        dataset=dataset,
        size=size,
        operations=operations,
        threads=threads
    )
    
    # Check if lidx directory exists
    lidx_dir = os.path.join("benchmarks", "lidx")
    if not os.path.exists(lidx_dir):
        print(f"Error: LIDX directory not found at {lidx_dir}")
        return
    
    # Check if benchmark script exists
    benchmark_script = os.path.join(lidx_dir, "run_lidx_benchmark.py")
    if not os.path.exists(benchmark_script):
        print(f"Error: LIDX benchmark script not found at {benchmark_script}")
        return
    
    # Run the benchmark script with global configuration
    print("Starting LIDX benchmark...")
    cmd = f"cd {lidx_dir} && python3 run_lidx_benchmark.py --drift {drift} --dataset {dataset} --size {size} --index {index_name} --operations {operations} --threads {threads} --verbose"
    result = os.system(cmd)
    
    if result == 0:
        print("[SUCCESS] LIDX benchmark completed successfully!")
    else:
        print(f"[FAILED] LIDX benchmark failed with exit code {result}")


def handle_lcc(tokens: List[str]):
    """Handle learned concurrency control command - test Polyjuice"""
    tokens = tokens[1:]
    
    if len(tokens) > 0:
        print("Error: LCC command takes no parameters")
        print("Usage: lcc")
        print("This will test Polyjuice using training scripts")
        return
    
    print("Testing Polyjuice (Learned Concurrency Control)...")
    print("Current global settings:")
    print(f"  Dataset: {GLOBAL_CONFIG['dataset']}")
    print(f"  Drift: {GLOBAL_CONFIG['drift']}")
    
    # Check if lcc directory exists
    lcc_dir = os.path.join("benchmarks", "lcc")
    if not os.path.exists(lcc_dir):
        print(f"Error: LCC directory not found at {lcc_dir}")
        return
    
    # Check if training directory exists
    training_dir = os.path.join(lcc_dir, "training")
    if not os.path.exists(training_dir):
        print(f"Error: LCC training directory not found at {training_dir}")
        return
    
    # Check if ERL training script exists
    erl_script = os.path.join(training_dir, "ERL_main.py")
    if not os.path.exists(erl_script):
        print(f"Error: ERL training script not found at {erl_script}")
        return
    
    # Use global configuration for scale factor (convert drift to scale)
    scale_factor = max(1, int(GLOBAL_CONFIG["drift"] * 10))
    
    print("Starting Polyjuice test with ERL training...")
    print(f"Scale factor: {scale_factor}")
    
    # Run the ERL training script with minimal parameters
    cmd = f"cd {training_dir} && python3 ERL_main.py --workload-type tpcc --scale-factor {scale_factor} --nworkers 8 --eval-time 1.0 --max-iterations 10 --samples-per-distribution 8 --psize 4"
    result = os.system(cmd)
    
    if result == 0:
        print("[SUCCESS] Polyjuice test completed successfully!")
    else:
        print(f"[FAILED] Polyjuice test failed with exit code {result}")
        print("Note: This is normal if LCC is not built yet.")
        print("To build LCC, run: cd benchmarks/lcc && MODE=perf make -j dbtest")


def regenerate_with_best_cached_params(dataset_name: str, table_name: str, force_if_not_validated: bool = True, skip_freq_preservation: bool = False, reference_dataset: str = None, variant_id: int = -1) -> bool:
    """
    Check if cache has better params than last generation. If so, regenerate.

    Args:
        force_if_not_validated: If True, force regenerate when validation_passed is False or missing.
                                This ensures we always use best params for tables that haven't passed validation.
        skip_freq_preservation: If True, skip frequency preservation (for non-drift-ref mode).
        reference_dataset: Reference dataset name for path construction.
        variant_id: Variant ID for path construction.

    Returns True if regeneration was done, False otherwise.
    """
    import json
    import subprocess

    metadata_path = os.path.join(get_expdir_path(dataset_name, table_name, reference_dataset, variant_id), "last_generation.json")
    if not os.path.exists(metadata_path):
        print(f"  No generation metadata found, will validate current data")
        return False

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)

        cache_path = metadata.get("cache_path")
        cache_key = metadata.get("cache_key")
        last_scale_factor = metadata.get("scale_factor")
        reference_dataset = metadata.get("reference_dataset")
        target_drift = metadata.get("target_drift")

        if not cache_path or not os.path.exists(cache_path):
            print(f"  No cache found, will validate current data")
            return False

        # Load cache
        with open(cache_path) as f:
            cache_data = json.load(f)

        if cache_key not in cache_data.get("best_params", {}):
            print(f"  No cached result for target_drift={cache_key}, will validate current data")
            return False

        cached_result = cache_data["best_params"][cache_key]
        cached_params = cached_result["params"]

        # Check if this was generated with a fallback method (year_offset, find_q, row_mixing)
        # Fallback results are deterministic and don't need regeneration via dbproc.py
        fallback_type = cached_result.get("fallback_type", "")
        if fallback_type:
            print(f"  Generated with {fallback_type} fallback, skipping regeneration")
            return False

        # Get cached scale_factor
        cached_scale_factor = cached_params.get("scale_factor", 8.0)

        # ALWAYS check scale_factor first - if it matches, no need to regenerate
        # This prevents double-generation when auto_tune.py just generated with best params
        if last_scale_factor is not None and abs(cached_scale_factor - last_scale_factor) < 0.01:
            print(f"  Last generation used best cached params (scale_factor={cached_scale_factor:.2f})")
            return False

        # Check validation status - if not validated and scale_factor differs, regenerate
        validation_passed = cached_result.get("validation_passed", False)
        if not validation_passed and force_if_not_validated:
            print(f"  Validation not passed and params differ, regenerating with best params")

        # Get all cached params with defaults for regeneration
        cached_scale_factor = cached_params.get("scale_factor", 8.0)
        cached_controller_dim = cached_params.get("controller_dim", [512, 512])
        cached_controller_steps = cached_params.get("controller_steps", 10000)
        cached_controller_bs = cached_params.get("controller_bs", 512)
        cached_controller_lr = cached_params.get("controller_lr", 0.001)
        cached_diffuser_steps = cached_params.get("diffuser_steps", 30000)
        cached_diffuser_bs = cached_params.get("diffuser_bs", 2048)
        cached_diffuser_lr = cached_params.get("diffuser_lr", 0.0018)
        cached_diffuser_timesteps = cached_params.get("diffuser_timesteps", 1000)
        cached_lambda_p = cached_params.get("lambda_p", 1.0)
        cached_lambda_s = cached_params.get("lambda_s", 1.0)

        # Convert dim to list if needed
        if isinstance(cached_controller_dim, tuple):
            cached_controller_dim = list(cached_controller_dim)

        # Need to regenerate with best cached params
        print(f"\n  Last generation: scale_factor={last_scale_factor:.2f}")
        print(f"  Best cached:     scale_factor={cached_scale_factor:.2f}, dim={cached_controller_dim}, steps={cached_controller_steps}")
        print(f"  Regenerating with retrain using best cached params...")

        # Build command with all cached params
        cmd = f"python3 -u dbproc.py --dataset-name={dataset_name} --table-name={table_name}"
        cmd += f" --drift={target_drift}"
        cmd += f" --scale-factor={cached_scale_factor}"
        # Controller params
        cmd += f" --controller-dim {' '.join(str(d) for d in cached_controller_dim)}"
        cmd += f" --controller-steps={cached_controller_steps}"
        cmd += f" --controller-bs={cached_controller_bs}"
        cmd += f" --controller-lr={cached_controller_lr}"
        # Diffuser params
        cmd += f" --diffuser-steps={cached_diffuser_steps}"
        cmd += f" --diffuser-bs={cached_diffuser_bs}"
        cmd += f" --diffuser-lr={cached_diffuser_lr}"
        cmd += f" --diffuser-timesteps={cached_diffuser_timesteps}"
        # Lambda params
        cmd += f" --lambda-p={cached_lambda_p}"
        cmd += f" --lambda-s={cached_lambda_s}"
        # Force retrain
        cmd += " --retrain-diffuser --retrain-controller"

        if reference_dataset:
            cmd += f" --reference-dataset={reference_dataset}"

        if skip_freq_preservation:
            cmd += " --skip-freq-preservation"

        print(f"  Running: {cmd}")

        result = subprocess.run(cmd, shell=True, capture_output=False, text=True, timeout=3600)

        if result.returncode != 0:
            print(f"  Regeneration failed!")
            return False

        # Update last_generation.json with the new params
        metadata["scale_factor"] = cached_scale_factor
        metadata["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Regeneration complete!")
        return True

    except Exception as e:
        print(f"  Error checking cache: {e}")
        return False


def update_cache_validation(dataset_name: str, table_name: str, passed: bool, time_ratio: float, reference_dataset: str = None, variant_id: int = -1):
    """Update tuning cache with validation result based on last_generation.json metadata."""
    import json

    # Check for last_generation.json
    metadata_path = os.path.join(get_expdir_path(dataset_name, table_name, reference_dataset, variant_id), "last_generation.json")
    if not os.path.exists(metadata_path):
        print(f"  No generation metadata found at {metadata_path}")
        return False

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)

        cache_path = metadata.get("cache_path")
        cache_key = metadata.get("cache_key")

        if not cache_path or not cache_key:
            print(f"  Metadata missing cache_path or cache_key")
            return False

        if not os.path.exists(cache_path):
            print(f"  Cache file not found: {cache_path}")
            return False

        # Load cache
        with open(cache_path) as f:
            cache_data = json.load(f)

        # Update validation status
        if cache_key in cache_data.get("best_params", {}):
            cache_data["best_params"][cache_key]["validation_passed"] = passed
            cache_data["best_params"][cache_key]["validation_ratio"] = time_ratio

            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)

            print(f"  Updated cache: {cache_path} (key={cache_key}, passed={passed}, ratio={time_ratio:.2f})")
            return True
        else:
            print(f"  Cache key {cache_key} not found in cache")
            return False

    except Exception as e:
        print(f"  Error updating cache: {e}")
        return False


def handle_validate_data(tokens: List[str]):
    """Handle validate data command - compare generated data with real database"""
    tokens = tokens[1:]

    if len(tokens) < 1:
        print("Error: Not enough arguments")
        print("Usage: vd DATASET [TABLE] [--gen-db=X] [--real-db=Y] [--threshold=N] [--force-reload] [--dry-run]")
        print("  --gen-db=X: Target database for generated data (default: imdb_17v2_gen)")
        print("  --real-db=Y: Reference real database (default: imdb_17v2)")
        print("  --threshold=N: Max execution time ratio (default: 1.2 = 20%)")
        print("  --force-reload: Force reload database even if exists")
        print("  --dry-run: Test without replacing tables (compare current db state)")
        print("Examples:")
        print("  vd imdb                           # Validate all tables")
        print("  vd imdb title                     # Validate only title table")
        print("  vd imdb --gen-db=imdb_gen --real-db=imdb_real")
        print("  vd imdb --force-reload            # Force reload database")
        print("  vd imdb --dry-run                 # Test current db state")
        return

    dataset_name = tokens[0]
    table_name = None
    gen_db = "imdb_17v2_gen"
    real_db = "imdb_17v2"
    threshold = 1.2  # 20% performance threshold (same as gd --validate)
    force_reload = False
    dry_run = False
    reference_dataset = None
    variant_id = -1

    # Parse flags
    remaining_tokens = []
    for t in tokens[1:]:
        if t.startswith("--gen-db="):
            gen_db = t[9:]
        elif t.startswith("--real-db="):
            real_db = t[10:]
        elif t.startswith("--threshold="):
            threshold = float(t[12:])
        elif t.startswith("--ref="):
            reference_dataset = t[6:]
        elif t.startswith("--variant-id="):
            variant_id = int(t[13:])
        elif t == "--force-reload":
            force_reload = True
        elif t == "--dry-run":
            dry_run = True
        else:
            remaining_tokens.append(t)

    if remaining_tokens:
        table_name = remaining_tokens[0]

    print_args(
        dataset_name=dataset_name,
        table_name=table_name or "(all tables)",
        gen_db=gen_db,
        real_db=real_db,
        threshold=f"{threshold}x",
        force_reload=force_reload,
        dry_run=dry_run,
        pg_port=GLOBAL_CONFIG["pg_port"],
    )

    # Import and use DatabaseValidator
    from db_validation import DatabaseValidator

    validator = DatabaseValidator(
        gen_db=gen_db,
        real_db=real_db,
        port=GLOBAL_CONFIG["pg_port"],
    )

    if table_name:
        # Check if we need to regenerate with best cached params
        if not dry_run:
            print(f"\nChecking for best cached params...")
            regenerate_with_best_cached_params(dataset_name, table_name, skip_freq_preservation=False, reference_dataset=reference_dataset, variant_id=variant_id)

        # Validate single table - setup database if not exists
        if validator.database_exists(gen_db):
            if force_reload:
                print(f"\nDatabase {gen_db} exists, force reloading...")
                validator.create_database(gen_db, force=True)
                validator.copy_all_tables(real_db, gen_db)
            else:
                print(f"\nDatabase {gen_db} already exists, skipping setup.")
        else:
            print(f"\nSetting up {gen_db} database...")
            validator.create_database(gen_db)
            validator.copy_all_tables(real_db, gen_db)

        result = validator.validate_single_table(dataset_name, table_name, threshold, dry_run=dry_run,
                                                  reference_dataset=reference_dataset, variant_id=variant_id)

        if result.passed:
            print(f"\n[SUCCESS] Table {table_name} passed validation!")
            # Update cache with validation result (skip in dry_run mode)
            if not dry_run:
                update_cache_validation(dataset_name, table_name, True, result.time_ratio, reference_dataset=reference_dataset, variant_id=variant_id)
        elif result.import_failed:
            print(f"\n[IMPORT FAILED] Table {table_name} - data import error, check CSV format")
        else:
            print(f"\n[FAILED] Table {table_name} failed validation (time_ratio={result.time_ratio:.2f}x)")
            # Update cache with failed validation (skip in dry_run mode)
            if not dry_run:
                update_cache_validation(dataset_name, table_name, False, result.time_ratio, reference_dataset=reference_dataset, variant_id=variant_id)
    else:
        # Get list of driftable tables
        info_path = os.path.join("datasets", dataset_name, "dataset_info.json")
        if os.path.exists(info_path):
            import json
            with open(info_path) as f:
                dataset_info = json.load(f)
            driftable_tables = [k for k, v in dataset_info.items() if v is not None]

            # Check and regenerate each table with best cached params (skip in dry_run mode)
            if not dry_run:
                print(f"\nChecking for best cached params for {len(driftable_tables)} tables...")
                for tbl in driftable_tables:
                    regenerate_with_best_cached_params(dataset_name, tbl, skip_freq_preservation=False, reference_dataset=reference_dataset, variant_id=variant_id)

        # Validate all tables
        results = validator.validate_all_tables(dataset_name, threshold, force_reload=force_reload, dry_run=dry_run,
                                                reference_dataset=reference_dataset, variant_id=variant_id)
        failed = [t for t, r in results.items() if not r.passed and not r.import_failed]
        import_failed = [t for t, r in results.items() if r.import_failed]

        # Update cache for each table (skip in dry_run mode)
        if not dry_run:
            print("\nUpdating tuning cache...")
            for tbl, result in results.items():
                if not result.import_failed:
                    update_cache_validation(dataset_name, tbl, result.passed, result.time_ratio, reference_dataset=reference_dataset, variant_id=variant_id)

        if len(failed) == 0 and len(import_failed) == 0:
            print("\n[SUCCESS] All tables passed validation!")
        else:
            if import_failed:
                print(f"\n[IMPORT FAILED] {len(import_failed)} table(s) - data import error: {import_failed}")
            if failed:
                print(f"\n[FAILED] {len(failed)} table(s) failed validation: {failed}")


def main():
    session = PromptSession(
        history=FileHistory(os.path.join(os.path.dirname(__file__), ".cli_history"))
    )

    while True:
        try:
            text = session.prompt("[NRBench]> ")
            text = text.strip()

        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        else:
            if text in ["q", "quit"]:
                break

            elif text in ["h", "help"]:
                show_help()
                continue

            if not text:
                continue

            tokens = text.split()
            if not tokens:
                continue

            main_command = tokens[0]

            if main_command == "gd":
                # Generate data command
                handle_generate_data(tokens)
                continue

            if main_command == "gq":
                # Generate query command
                print("Error: Generate query command is not yet implemented.")
                continue

            if main_command == "dd":
                handle_delete_data(tokens)
                continue

            if main_command == "tqo":
                # Train learned query optimizer command
                handle_tqo(tokens)
                continue

            if main_command == "iqo":
                # Inference learned query optimizer command
                handle_iqo(tokens)
                continue

            if main_command == "set":
                # Set global configuration command
                handle_set(tokens)
                continue
                
            if main_command == "idx":
                # Test learned index command
                handle_idx(tokens)
                continue
                
            if main_command == "lcc":
                # Test learned concurrency control command
                handle_lcc(tokens)
                continue

            if main_command == "vd":
                # Validate generated data command
                handle_validate_data(tokens)
                continue

            print("Unknown command:", text)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Command-line mode: run command directly and exit
        text = " ".join(sys.argv[1:])
        tokens = text.split()
        main_command = tokens[0]

        if main_command == "gd":
            handle_generate_data(tokens)
        elif main_command == "vd":
            handle_validate_data(tokens)
        elif main_command == "dd":
            handle_delete_data(tokens)
        elif main_command == "tqo":
            handle_tqo(tokens)
        elif main_command == "iqo":
            handle_iqo(tokens)
        elif main_command == "set":
            handle_set(tokens)
        elif main_command == "idx":
            handle_idx(tokens)
        elif main_command == "lcc":
            handle_lcc(tokens)
        else:
            print("Unknown command:", main_command)
    else:
        # Interactive mode
        main()
