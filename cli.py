import json
import os
import sys
import shutil
import time
import subprocess
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

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
  gd DATASET [TABLE] DRIFT [--auto] [--quick] [--gpus=0,1,2] [--validate]
                                     Generate data that drifts DRIFT on DATASET
                                       --auto: Auto-tune parameters for target drift
                                       --quick: Quick tune (reuse existing models)
                                       --gpus=X,Y,Z: Use multiple GPUs for parallel execution
                                                     (only when TABLE is not specified)
                                       --validate: Validate with DB after generation, re-tune if needed
                                                   (pass if ratio in [1/1.2, 1.2] = within ±20%)
                                       --dry-run: Test validation without replacing tables
  gd DATASET --drift-ref=FILE [--gpus=0,1,2] [--validate] [--dry-run]
                                     Generate data using drift reference file
                                       The reference file should contain per-table drift values
                                       (e.g., from calc_drift.py --src-dir X --dst-dir Y)
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
    quick_tune: bool,
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

            if quick_tune:
                tune_result = tuner.quick_tune(drift)
            else:
                tune_result = tuner.tune(drift, max_iterations=100, tolerance=0.20)

            if tune_result:
                result["success"] = True
                result["message"] = f"drift_error={tune_result.drift_error:.4f}, corr_loss={tune_result.correlation_loss:.4f}"
            else:
                result["message"] = "Auto-tuning failed"
        else:
            # Check for cached params first
            cached_params = get_cached_params_for_table(dataset_name, table_name, drift, reference_dataset)
            if cached_params:
                cmd = f"python3 dbproc.py --dataset-name={dataset_name} --table-name={table_name} --drift={drift}"
                cmd += f" --device={device} {cached_params.to_cmd_args()} --reuse{ref_arg}"
            else:
                cmd = f"python3 dbproc.py --dataset-name={dataset_name} --table-name={table_name} --drift={drift}"
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


def handle_generate_data(tokens: List[str]):
    tokens = tokens[1:]

    if len(tokens) < 1:
        print("Error: Not enough arguments")
        print("Usage: gd DATASET [TABLE] DRIFT [--auto] [--quick] [--ref=REFERENCE_DATASET] [--gpus=0,1,2]")
        print("       gd DATASET --drift-ref=FILE [--gpus=0,1,2]")
        print("  --auto: Enable auto-tuning to find best parameters")
        print("  --quick: Quick auto-tune (uses existing models, faster)")
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
    quick_tune = False
    reference_dataset = None
    gpus = None  # List of GPU IDs for parallel execution
    drift_ref_file = None  # Drift reference file path
    validate_after = False  # Whether to validate with DB after generation
    validate_threshold = 1.2  # 20% performance threshold
    dry_run = False  # Dry run: test without replacing tables

    # Parse flags
    remaining_tokens = []
    for t in tokens[1:]:
        if t == "--auto":
            auto_tune = True
        elif t == "--quick":
            quick_tune = True
            auto_tune = True  # quick implies auto
        elif t == "--validate":
            validate_after = True
        elif t == "--dry-run":
            dry_run = True
            validate_after = True  # dry-run implies validate
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
            dataset_name, drift_ref_file, reference_dataset, quick_tune, gpus,
            single_table=specified_table,
            validate_after=validate_after,
            validate_threshold=validate_threshold,
            dry_run=dry_run,
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
        print_args(
            dataset_name=dataset_name,
            table_name=table_name,
            drift=drift,
            scale=scale,
            auto_tune=auto_tune,
            quick_tune=quick_tune,
            reference_dataset=reference_dataset or f"{dataset_name}_2014 (default)",
            device=device,
        )

        if auto_tune:
            # Non drift-ref mode: ignore --validate (no real data to validate against)
            _generate_with_auto_tune(dataset_name, table_name, drift, quick_tune, reference_dataset, device, require_validation=False)
        else:
            # Check for cached params first
            cached_params = get_cached_params_for_table(dataset_name, table_name, drift, reference_dataset)
            if cached_params:
                print(f"Using cached parameters: scale_factor={cached_params.scale_factor}")
                cmd = f"python3 dbproc.py --dataset-name={dataset_name} --table-name={table_name} --drift={drift}"
                cmd += f" --device={device} {cached_params.to_cmd_args()} --reuse{ref_arg}"
                os.system(cmd)
            else:
                os.system(
                    f"python3 dbproc.py --dataset-name={dataset_name} --table-name={table_name} --drift={drift} --device={device} --scale-factor={scale}{ref_arg}"
                )
    else:
        # run on all tables
        ## get table names from dataset_info.json
        base_dir = os.path.join("datasets", dataset_name)
        config: dict = load_json(os.path.join(base_dir, "dataset_info.json"))
        table_names = [t for t in config.keys() if config[t]]
        print_args(
            dataset_name=dataset_name,
            table_names=table_names,
            drift=drift,
            scale=scale,
            auto_tune=auto_tune,
            quick_tune=quick_tune,
            reference_dataset=reference_dataset or f"{dataset_name}_2014 (default)",
            gpus=gpus if gpus else [0],
        )

        if gpus and len(gpus) > 1:
            # Multi-GPU parallel execution
            print(f"\n{'='*60}")
            print(f"Running parallel generation on GPUs: {gpus}")
            print(f"Tables: {table_names}")
            print(f"{'='*60}\n")

            # Distribute tables across GPUs using subprocess
            processes = []
            table_gpu_assignments = []

            for i, t in enumerate(table_names):
                gpu_id = gpus[i % len(gpus)]
                table_gpu_assignments.append((t, gpu_id))

            print("Table -> GPU assignments:")
            for t, gpu_id in table_gpu_assignments:
                print(f"  {t} -> GPU {gpu_id}")
            print()

            # Launch all processes
            for t, gpu_id in table_gpu_assignments:
                ref_arg = f" --reference-dataset={reference_dataset}" if reference_dataset else ""

                if auto_tune:
                    # Use auto_tune.py script directly for better subprocess handling
                    # Non drift-ref mode: ignore --validate (no real data to validate against)
                    cmd = f"python3 auto_tune.py --dataset-name={dataset_name} --table-name={t}"
                    cmd += f" --target-drift={drift} --device={gpu_id}"
                    if reference_dataset:
                        cmd += f" --reference-dataset={reference_dataset}"
                    if quick_tune:
                        cmd += " --quick"
                else:
                    cached_params = get_cached_params_for_table(dataset_name, t, drift, reference_dataset)
                    if cached_params:
                        cmd = f"python3 dbproc.py --dataset-name={dataset_name} --table-name={t} --drift={drift}"
                        cmd += f" --device={gpu_id} {cached_params.to_cmd_args()} --reuse{ref_arg}"
                    else:
                        cmd = f"python3 dbproc.py --dataset-name={dataset_name} --table-name={t} --drift={drift}"
                        cmd += f" --device={gpu_id} --scale-factor={scale}{ref_arg}"

                print(f"[GPU {gpu_id}] Starting: {t}")

                # Create log file for each table
                log_dir = "gd_logs"
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"{dataset_name}_{t}_{drift}.log")

                with open(log_file, "w") as f:
                    proc = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                    )
                    processes.append((t, gpu_id, proc, log_file))

            # Wait for all processes to complete
            print(f"\nWaiting for {len(processes)} processes to complete...")
            print(f"Logs are being written to: {log_dir}/")
            print()

            completed = 0
            failed = []
            for t, gpu_id, proc, log_file in processes:
                ret = proc.wait()
                completed += 1
                if ret == 0:
                    print(f"[{completed}/{len(processes)}] ✓ {t} (GPU {gpu_id}) completed successfully")
                else:
                    print(f"[{completed}/{len(processes)}] ✗ {t} (GPU {gpu_id}) failed (exit code {ret})")
                    print(f"    Check log: {log_file}")
                    failed.append(t)

            print(f"\n{'='*60}")
            print(f"Parallel generation complete: {len(processes) - len(failed)}/{len(processes)} succeeded")
            if failed:
                print(f"Failed tables: {failed}")
            print(f"{'='*60}")

        else:
            # Sequential execution (original behavior)
            device = gpus[0] if gpus else 0

            for t in table_names:
                print(f"\n{'='*60}")
                print(f"Generating data for {dataset_name}.{t} on GPU {device}...")
                print(f"{'='*60}")

                if auto_tune:
                    # Non drift-ref mode: ignore --validate (no real data to validate against)
                    _generate_with_auto_tune(dataset_name, t, drift, quick_tune, reference_dataset, device, require_validation=False)
                else:
                    # Check for cached params first
                    cached_params = get_cached_params_for_table(dataset_name, t, drift, reference_dataset)
                    if cached_params:
                        print(f"Using cached parameters: scale_factor={cached_params.scale_factor}")
                        cmd = f"python3 dbproc.py --dataset-name={dataset_name} --table-name={t} --drift={drift}"
                        cmd += f" --device={device} {cached_params.to_cmd_args()} --reuse{ref_arg}"
                        os.system(cmd)
                    else:
                        os.system(
                            f"python3 dbproc.py --dataset-name={dataset_name} --table-name={t} --drift={drift} --device={device} --scale-factor={scale}{ref_arg}"
                        )

                print(f"Data generation complete for {dataset_name}.{t}")

    print("\nData generation complete")


def _generate_with_auto_tune(
    dataset_name: str,
    table_name: str,
    drift: float,
    quick: bool = False,
    reference_dataset: str = None,
    device: int = 0,
    target_corr_loss: float = None,
    no_cache: bool = False,
    require_validation: bool = False,
) -> bool:
    """Generate data with automatic parameter tuning.

    Returns:
        True if used validated cache (skip validation), False otherwise
    """
    corr_str = f", target_corr={target_corr_loss:.4f}" if target_corr_loss is not None else ""
    cache_str = " [NO CACHE]" if no_cache else ""
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
    )

    if quick:
        result = tuner.quick_tune(drift, use_cache=not no_cache)
    else:
        result = tuner.tune(drift, max_iterations=100, tolerance=0.20, use_cache=not no_cache, require_validation=require_validation)

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
            print(f"  Corr error: {corr_error:.4f} (tolerance: 0.20)")
        print(f"  Best scale_factor: {result.params.scale_factor}")

        # Check if this was a validated cache hit
        if result.validation_passed:
            used_validated_cache = True
            print(f"  Using validated cache (skip validation)")

        # Ensure we have data generated with best params
        output_path = os.path.join("expdir", dataset_name, table_name, f"{table_name}.drifted.csv")
        if not os.path.exists(output_path):
            # File doesn't exist, generate with best cached params
            print(f"\nGenerating data with cached parameters...")
            tuner.generate_with_best_params(drift, use_cache=True)
        else:
            # File exists, check if it was generated with best params
            # (last iteration might not be the best one)
            regenerate_with_best_cached_params(dataset_name, table_name)
    else:
        print(f"\nAuto-tuning failed. Using default parameters.")
        ref_arg = f" --reference-dataset={reference_dataset}" if reference_dataset else ""
        os.system(
            f"python3 dbproc.py --dataset-name={dataset_name} --table-name={table_name} --drift={drift}{ref_arg}"
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
    dry_run: bool = False,
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

    print(f"\nValidating {len(table_names)} table(s) against database...")
    print(f"  gen_db: {gen_db}")
    print(f"  real_db: {real_db}")
    print(f"  threshold: {threshold:.1%}")
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

    for table_name in table_names:
        print(f"\n--- Validating: {table_name} ---")

        # Ensure we have data generated with best cached params
        if not dry_run:
            regenerate_with_best_cached_params(dataset_name, table_name)

        result = validator.validate_single_table(dataset_name, table_name, threshold, dry_run=dry_run)

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


def _generate_with_drift_reference(
    dataset_name: str,
    drift_ref_file: str,
    reference_dataset: Optional[str],
    quick_tune: bool,
    gpus: Optional[List[int]],
    single_table: Optional[str] = None,
    validate_after: bool = False,
    validate_threshold: float = 1.2,
    max_retries: int = 3,
    dry_run: bool = False,
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
    if validate_after:
        if dry_run:
            print(f"Validation: DRY RUN (no table replacement, threshold={validate_threshold:.1%})")
        else:
            print(f"Validation: enabled (threshold={validate_threshold:.1%})")
    print()

    # Print per-table drift and correlation targets
    print("Target per table:")
    for table, (target_drift, target_corr) in drift_ref.items():
        corr_str = f"{target_corr:.4f}" if target_corr is not None else "N/A"
        print(f"  {table}: drift={target_drift:.4f}, corr_loss={corr_str}")
    print()

    # Track tables that need to be generated
    tables_to_generate = dict(drift_ref)
    generated_tables = set()
    validated_cache_tables = set()  # Tables that used validated cache (skip validation)
    retry_counts = {t: 0 for t in drift_ref.keys()}
    force_retrain_tables = set()  # Tables that need forced retrain (validation failed)

    while tables_to_generate:
        if gpus and len(gpus) > 1:
            # Multi-GPU parallel execution
            print(f"Running parallel generation on GPUs: {gpus}")
            print()

            # Distribute tables across GPUs
            table_gpu_assignments = []
            table_names = list(tables_to_generate.keys())
            for i, t in enumerate(table_names):
                gpu_id = gpus[i % len(gpus)]
                target_drift, target_corr = tables_to_generate[t]
                table_gpu_assignments.append((t, gpu_id, target_drift, target_corr))

            print("Table -> GPU assignments:")
            for t, gpu_id, target_drift, target_corr in table_gpu_assignments:
                corr_str = f"{target_corr:.4f}" if target_corr is not None else "N/A"
                print(f"  {t} (drift={target_drift:.4f}, corr={corr_str}) -> GPU {gpu_id}")
            print()

            # Launch all processes
            processes = []
            log_files = []  # Keep file handles open until all processes complete
            log_dir = "gd_logs"
            os.makedirs(log_dir, exist_ok=True)

            for t, gpu_id, target_drift, target_corr in table_gpu_assignments:
                cmd = f"python3 auto_tune.py --dataset-name={dataset_name} --table-name={t}"
                cmd += f" --target-drift={target_drift} --device={gpu_id}"
                # Note: Don't pass target_corr_loss - with new logic we directly compare
                # generated vs reference correlation, so we want to minimize it (target=0)
                if reference_dataset:
                    cmd += f" --reference-dataset={reference_dataset}"
                if quick_tune:
                    cmd += " --quick"
                if t in force_retrain_tables:
                    cmd += " --no-cache"  # Force retrain for validation-failed tables
                # Note: --require-validation is deprecated, cache is used if it meets tolerance
                # Validation happens later for tables without validation_passed=True

                corr_str = f", corr={target_corr:.4f}" if target_corr is not None else ""
                retry_str = f" [RETRY {retry_counts[t]}]" if t in force_retrain_tables else ""
                print(f"[GPU {gpu_id}] Starting: {t} (drift={target_drift:.4f}{corr_str}){retry_str}")

                log_file_path = os.path.join(log_dir, f"{dataset_name}_{t}_ref.log")
                log_file = open(log_file_path, "w", buffering=1)  # Line buffering
                log_files.append(log_file)  # Keep handle open

                # Use unbuffered Python output for real-time logging
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"

                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                processes.append((t, gpu_id, target_drift, target_corr, proc, log_file_path))

            # Wait for all processes to complete with periodic status updates
            print(f"\nWaiting for {len(processes)} processes to complete...")
            print(f"Logs are being written to: {log_dir}/")
            print(f"Status updates every 30 seconds. Use 'tail -f {log_dir}/*.log' for real-time logs.")
            print()

            def get_log_status(log_path):
                """Get last meaningful line from log file."""
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        # Find last line with iteration or result info
                        for line in reversed(lines[-20:]):
                            line = line.strip()
                            if any(kw in line for kw in ['Iteration', 'drift_error', 'Score', 'Drift error', 'Best', 'completed']):
                                return line[:80]  # Truncate long lines
                        if lines:
                            return lines[-1].strip()[:80]
                except:
                    pass
                return "starting..."

            # Poll until all complete
            gen_failed = []
            completed = set()
            last_status_time = time.time()
            status_interval = 30  # seconds

            while len(completed) < len(processes):
                # Check for completed processes
                for t, gpu_id, target_drift, target_corr, proc, log_file in processes:
                    if t in completed:
                        continue
                    ret = proc.poll()
                    if ret is not None:
                        completed.add(t)
                        corr_str = f", corr={target_corr:.4f}" if target_corr is not None else ""
                        if ret == 0:
                            print(f"✓ {t} (GPU {gpu_id}, drift={target_drift:.4f}{corr_str}) completed")
                            generated_tables.add(t)
                        elif ret == 2:
                            print(f"✓ {t} (GPU {gpu_id}, drift={target_drift:.4f}{corr_str}) completed (validated cache)")
                            generated_tables.add(t)
                            validated_cache_tables.add(t)
                        else:
                            print(f"✗ {t} (GPU {gpu_id}) failed (exit code {ret})")
                            print(f"    Check log: {log_file}")
                            gen_failed.append(t)

                # Periodic status update
                if time.time() - last_status_time >= status_interval and len(completed) < len(processes):
                    last_status_time = time.time()
                    print(f"\n--- Status Update ({len(completed)}/{len(processes)} completed) ---")
                    for t, gpu_id, target_drift, target_corr, proc, log_file in processes:
                        if t not in completed:
                            status = get_log_status(log_file)
                            print(f"  [GPU {gpu_id}] {t}: {status}")
                    print()

                time.sleep(1)  # Check every second

            # Close all log file handles
            for f in log_files:
                f.close()

            print(f"\nGeneration: {len(generated_tables)}/{len(drift_ref)} succeeded")
            if validated_cache_tables:
                print(f"Using validated cache (skip validation): {list(validated_cache_tables)}")
            if gen_failed:
                print(f"Failed: {gen_failed}")

        else:
            # Sequential execution
            device = gpus[0] if gpus else 0

            for table_name, (target_drift, target_corr) in tables_to_generate.items():
                corr_str = f", corr_loss={target_corr:.4f}" if target_corr is not None else ""
                retry_str = f" [RETRY {retry_counts[table_name]}]" if table_name in force_retrain_tables else ""
                print(f"\n{'='*60}")
                print(f"Generating: {table_name} (drift={target_drift:.4f}{corr_str}) on GPU {device}{retry_str}")
                print(f"{'='*60}")

                # Force no-cache for validation-failed tables
                no_cache = table_name in force_retrain_tables

                # Note: Don't pass target_corr_loss anymore - with new logic we directly
                # compare generated vs reference correlation, so we want to minimize it (target=0)
                used_validated_cache = _generate_with_auto_tune(
                    dataset_name, table_name, target_drift, quick_tune, reference_dataset, device,
                    target_corr_loss=None, no_cache=no_cache, require_validation=validate_after
                )
                generated_tables.add(table_name)
                if used_validated_cache:
                    validated_cache_tables.add(table_name)

        # Clear tables_to_generate for now (will re-add if validation fails)
        tables_to_generate = {}

        # Validation phase
        # Skip tables that used validated cache
        tables_to_validate = generated_tables - validated_cache_tables
        if validate_after and tables_to_validate:
            print(f"\n{'#'*60}")
            print(f"VALIDATION PHASE")
            print(f"{'#'*60}")
            if validated_cache_tables:
                print(f"Skipping validated cache tables: {list(validated_cache_tables)}")
            print(f"Tables to validate: {list(tables_to_validate)}")

            failed_tables, import_failed_tables = _validate_and_get_failed_tables(
                dataset_name, list(tables_to_validate), validate_threshold,
                drift_ref=drift_ref, reference_dataset=reference_dataset,
                dry_run=dry_run,
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
                print(f"\n✓ All {len(tables_to_validate)} table(s) passed validation!")
        elif validate_after and validated_cache_tables:
            print(f"\n{'#'*60}")
            print(f"VALIDATION PHASE")
            print(f"{'#'*60}")
            print(f"All tables used validated cache, skipping validation")

    print(f"\n{'='*60}")
    print(f"All tables processed using drift reference file")
    if validate_after:
        passed = len([t for t in drift_ref if retry_counts.get(t, 0) <= max_retries])
        print(f"Validation: {passed}/{len(drift_ref)} passed")
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


def regenerate_with_best_cached_params(dataset_name: str, table_name: str) -> bool:
    """
    Check if cache has better params than last generation. If so, regenerate.
    Returns True if regeneration was done, False otherwise.
    """
    import json
    import subprocess

    metadata_path = os.path.join("expdir", dataset_name, table_name, "last_generation.json")
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
        cached_scale_factor = cached_result["params"]["scale_factor"]

        # Compare scale_factor
        if abs(cached_scale_factor - last_scale_factor) < 0.01:
            print(f"  Last generation used best cached params (scale_factor={cached_scale_factor:.2f})")
            return False

        # Need to regenerate with best cached params
        print(f"\n  Last generation: scale_factor={last_scale_factor:.2f}")
        print(f"  Best cached:     scale_factor={cached_scale_factor:.2f}")
        print(f"  Regenerating with best cached params...")

        # Build command
        cmd = f"python3 dbproc.py --dataset-name={dataset_name} --table-name={table_name}"
        cmd += f" --drift={target_drift}"
        cmd += f" --scale-factor={cached_scale_factor}"
        cmd += " --reuse"  # Don't retrain, just regenerate

        if reference_dataset:
            cmd += f" --reference-dataset={reference_dataset}"

        print(f"  Running: {cmd}")

        result = subprocess.run(cmd, shell=True, capture_output=False, text=True, timeout=3600)

        if result.returncode != 0:
            print(f"  Regeneration failed!")
            return False

        # Update last_generation.json with the new scale_factor
        metadata["scale_factor"] = cached_scale_factor
        metadata["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Regeneration complete!")
        return True

    except Exception as e:
        print(f"  Error checking cache: {e}")
        return False


def update_cache_validation(dataset_name: str, table_name: str, passed: bool, time_ratio: float):
    """Update tuning cache with validation result based on last_generation.json metadata."""
    import json

    # Check for last_generation.json
    metadata_path = os.path.join("expdir", dataset_name, table_name, "last_generation.json")
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

    # Parse flags
    remaining_tokens = []
    for t in tokens[1:]:
        if t.startswith("--gen-db="):
            gen_db = t[9:]
        elif t.startswith("--real-db="):
            real_db = t[10:]
        elif t.startswith("--threshold="):
            threshold = float(t[12:])
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
            regenerate_with_best_cached_params(dataset_name, table_name)

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

        result = validator.validate_single_table(dataset_name, table_name, threshold, dry_run=dry_run)

        if result.passed:
            print(f"\n[SUCCESS] Table {table_name} passed validation!")
            # Update cache with validation result (skip in dry_run mode)
            if not dry_run:
                update_cache_validation(dataset_name, table_name, True, result.time_ratio)
        elif result.import_failed:
            print(f"\n[IMPORT FAILED] Table {table_name} - data import error, check CSV format")
        else:
            print(f"\n[FAILED] Table {table_name} failed validation (time_ratio={result.time_ratio:.2f}x)")
            # Update cache with failed validation (skip in dry_run mode)
            if not dry_run:
                update_cache_validation(dataset_name, table_name, False, result.time_ratio)
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
                    regenerate_with_best_cached_params(dataset_name, tbl)

        # Validate all tables
        results = validator.validate_all_tables(dataset_name, threshold, force_reload=force_reload, dry_run=dry_run)
        failed = [t for t, r in results.items() if not r.passed and not r.import_failed]
        import_failed = [t for t, r in results.items() if r.import_failed]

        # Update cache for each table (skip in dry_run mode)
        if not dry_run:
            print("\nUpdating tuning cache...")
            for tbl, result in results.items():
                if not result.import_failed:
                    update_cache_validation(dataset_name, tbl, result.passed, result.time_ratio)

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
    main()
