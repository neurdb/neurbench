#!/usr/bin/env python3
"""
Validation script for generated data experiments.

Task 1: Validate generated data against reference databases
Task 2: Train and test BAO on validated data
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TeeLogger:
    """Write output to both console and log file."""
    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Baseline databases
IMDB = "imdb"  # Original IMDB database (for drift comparison)
IMDB_2017 = "imdb_2017"  # IMDB 2017 database (for ref comparison)

# PostgreSQL config
PG_HOST = "172.28.176.110"
PG_PORT = 5430
PG_USER = "postgres"
PG_PASSWORD = "postgres"

# Validation threshold (bidirectional: ratio must be in [1/threshold, threshold])
DEFAULT_THRESHOLD = 1.2  # 20%

# Experiment configurations
# Task 1: base datasets → imdb_2017 (disabled)
# TASK_1_BASES = ["imdb", "imdb_2013", "imdb_2014", "imdb_2015", "imdb_2016"]
TASK_1_BASES = []
TASK_1_REF = "imdb_2017"

# Task 2: different drift values (no ref, use variant_id)
TASK_2_DRIFT_VALUES = [0.1, 0.3, 0.5]
TASK_2_BASE = "imdb"

# Task 3: base datasets → imdb_2015
# TASK_3_BASES = ["imdb", "imdb_2013", "imdb_2014"]
TASK_3_BASES = ["imdb_2013", "imdb_2014"]
TASK_3_REF = "imdb_2015"

# Tables to validate (exclude title for now)
DRIFTABLE_TABLES = ["aka_name", "aka_title", "movie_companies", "movie_keyword", "movie_link", "person_info"]

# BAO config
BAO_DIR = "benchmarks/lqos/bao"
JOB_QUERY_DIR = "datasets/workloads/bao/join-order-benchmark"

# Execution mode: "single" (prewarm + 1 run) or "triple" (3 runs, use 3rd)
EXECUTION_MODE = "single"

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    dataset_name: str
    reference_dataset: Optional[str]
    variant_id: int
    expdir_path: str
    gen_db_name: str
    compare_db: str  # Database to compare against
    threshold: float
    drift_value: Optional[float] = None  # For Task 2


@dataclass
class ValidationResult:
    """Result of validating a single table."""
    table_name: str
    passed: bool
    gen_time: float
    compare_time: float
    ratio: float
    error: Optional[str] = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def run_command(cmd: str, description: str = "", capture: bool = False, stream: bool = True) -> Tuple[bool, str]:
    """Run a shell command.

    Args:
        capture: If True, capture and return output
        stream: If True, show output in real-time (works with capture=True too)
    """
    print(f"\n>>> {description or cmd}", flush=True)

    if capture and stream:
        # Real-time output while also capturing
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        output_lines = []
        for line in process.stdout:
            print(line, end='', flush=True)
            output_lines.append(line)
        process.wait()
        return process.returncode == 0, ''.join(output_lines)
    elif capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout + result.stderr
    else:
        result = subprocess.run(cmd, shell=True)
        return result.returncode == 0, ""


def get_connection(dbname: str):
    """Get PostgreSQL connection."""
    import psycopg2
    conn_str = f"dbname={dbname} user={PG_USER} password={PG_PASSWORD} host={PG_HOST} port={PG_PORT}"
    return psycopg2.connect(conn_str)


def database_exists(dbname: str) -> bool:
    """Check if database exists."""
    try:
        conn = get_connection("postgres")
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        return result is not None
    except Exception as e:
        print(f"Error checking database: {e}")
        return False


def create_database_from(src_db: str, dst_db: str) -> bool:
    """Create database by copying from source."""
    print(f"\nCreating {dst_db} from {src_db}...")
    cmd = f"docker exec -i pg_bao bash -c \"su - postgres -c 'dropdb -p 5432 --if-exists {dst_db} && createdb -p 5432 {dst_db} && pg_dump -p 5432 {src_db} | psql -p 5432 {dst_db}'\""
    success, _ = run_command(cmd, f"Copy {src_db} -> {dst_db}")
    return success


def get_expdir_path(dataset_name: str, reference_dataset: str = None, variant_id: int = -1) -> str:
    """Get expdir path matching dbproc.py logic."""
    dataset_dir = dataset_name
    if reference_dataset and reference_dataset != dataset_name:
        dataset_dir = f"{dataset_name}_ref_{reference_dataset}"
    if variant_id > 0:
        dataset_dir += f"-{variant_id}"
    return os.path.join("expdir", dataset_dir)


def table_has_generated_data(expdir: str, table_name: str) -> bool:
    """Check if table has generated CSV file."""
    csv_path = os.path.join(expdir, table_name, f"{table_name}.drifted.csv")
    return os.path.exists(csv_path)


def get_baseline_performance(db_name: str, cache_file: str = None) -> Dict[str, float]:
    """
    Get baseline query performance for a database.
    Returns dict of {query_name: execution_time_ms}
    """
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached performance from {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    print(f"\nRunning JOB queries on {db_name} (mode: {EXECUTION_MODE})...")
    output_file = f"validation_logs/{db_name}_baseline.log"
    os.makedirs("validation_logs", exist_ok=True)

    # Run ANALYZE first
    try:
        conn = get_connection(db_name)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("ANALYZE VERBOSE")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Warning: ANALYZE failed: {e}")

    # Run queries with execution mode
    cmd = f"cd {BAO_DIR} && python3 run_test_queries.py " \
          f"--use_postgres --database_name {db_name} " \
          f"--query_dir ../../../{JOB_QUERY_DIR} " \
          f"--output_file ../../../{output_file} --db-port {PG_PORT} " \
          f"--execution-mode {EXECUTION_MODE}"
    success, _ = run_command(cmd, f"Run JOB queries on {db_name}")

    if not success:
        print(f"Failed to run queries on {db_name}")
        return {}

    # Parse results based on execution mode
    # single mode: use index 0, triple mode: use index 2
    result_index = 0 if EXECUTION_MODE == 'single' else 2
    results = {}
    with open(output_file) as f:
        for line in f:
            parts = line.strip().split(', ')
            if len(parts) >= 6:
                exec_idx = int(parts[1])
                if exec_idx != result_index:
                    continue
                query_path = parts[3]
                exec_time = float(parts[5])
                query_name = os.path.basename(query_path).replace('.sql', '')
                results[query_name] = exec_time

    # Cache results
    if cache_file:
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Cached performance to {cache_file}")

    return results


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

def get_all_experiments() -> List[ExperimentConfig]:
    """Get all experiment configurations."""
    experiments = []

    # Task 1: base → imdb_2017
    for base in TASK_1_BASES:
        exp = ExperimentConfig(
            name=f"task1_{base}_to_{TASK_1_REF}",
            dataset_name=base,
            reference_dataset=TASK_1_REF,
            variant_id=-1,
            expdir_path=get_expdir_path(base, TASK_1_REF),
            gen_db_name=f"{base}_ref_{TASK_1_REF}_gen",
            compare_db=IMDB_2017,
            threshold=DEFAULT_THRESHOLD,
        )
        experiments.append(exp)

    # Task 2: different drift values
    # threshold 是最低要求 (下界): gen_time / ori_time >= threshold
    # drift=0.1: ratio >= 1.5 (至少高50%)
    # drift=0.3: ratio >= 2.0 (至少高1倍)
    # drift=0.5: ratio >= 2.5 (至少高1.5倍)
    for drift_val in TASK_2_DRIFT_VALUES:
        variant_id = int(drift_val * 10)
        if drift_val <= 0.1:
            min_ratio = 1.5   # 至少高50%
        elif drift_val <= 0.3:
            min_ratio = 2.0   # 至少高1倍
        else:
            min_ratio = 2.5   # 至少高1.5倍

        exp = ExperimentConfig(
            name=f"task2_drift_{drift_val}",
            dataset_name=TASK_2_BASE,
            reference_dataset=None,
            variant_id=variant_id,
            expdir_path=get_expdir_path(TASK_2_BASE, None, variant_id),
            gen_db_name=f"imdb_drift_{variant_id}_gen",
            compare_db=IMDB,  # Compare against original
            threshold=min_ratio,  # 这里 threshold 作为下界使用
            drift_value=drift_val,
        )
        experiments.append(exp)

    # Task 3: base → imdb_2015
    # ref=imdb_2015, 和 imdb_2015 的数据库对比
    for base in TASK_3_BASES:
        exp = ExperimentConfig(
            name=f"task3_{base}_to_{TASK_3_REF}",
            dataset_name=base,
            reference_dataset=TASK_3_REF,
            variant_id=-1,
            expdir_path=get_expdir_path(base, TASK_3_REF),
            gen_db_name=f"{base}_ref_{TASK_3_REF}_gen",
            compare_db=TASK_3_REF,  # Compare against imdb_2015 database
            threshold=DEFAULT_THRESHOLD,
        )
        experiments.append(exp)

    return experiments


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_table(
    exp: ExperimentConfig,
    table_name: str,
    baseline_perf: Dict[str, float],
) -> ValidationResult:
    """Validate a single table for an experiment."""
    import re

    csv_path = os.path.join(exp.expdir_path, table_name, f"{table_name}.drifted.csv")
    if not os.path.exists(csv_path):
        return ValidationResult(
            table_name=table_name,
            passed=False,
            gen_time=0,
            compare_time=0,
            ratio=0,
            error=f"CSV not found: {csv_path}"
        )

    # For Task 2 (drift experiments without ref), use different validation:
    # - threshold is minimum ratio required (lower bound)
    # - ratio = gen_time / ori_time must be >= threshold
    is_drift_experiment = exp.drift_value is not None

    # Build vd command
    # For drift experiments, use high threshold (10.0) to always "pass" vd,
    # then check ratio ourselves
    cmd = f"python cli.py vd {exp.dataset_name} {table_name}"
    cmd += f" --gen-db={exp.gen_db_name}"
    cmd += f" --real-db={exp.compare_db}"
    if is_drift_experiment:
        cmd += f" --threshold=10.0"  # High threshold to get ratio without failing
    else:
        cmd += f" --threshold={exp.threshold}"
    if exp.reference_dataset:
        cmd += f" --ref={exp.reference_dataset}"
    if exp.variant_id > 0:
        cmd += f" --variant-id={exp.variant_id}"

    print(f"\n{'='*60}")
    print(f"Validating: {exp.name} / {table_name}")
    if is_drift_experiment:
        print(f"Drift experiment: ratio must be >= {exp.threshold:.2f}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    success, output = run_command(cmd, f"Validate {table_name}", capture=True)
    print(output)

    # Save output to per-task log file (append mode)
    log_dir = f"validation_logs/{exp.expdir_path.replace('expdir/', '')}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "validation.log")
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Table: {table_name}\n")
        f.write(f"Command: {cmd}\n")
        f.write(f"{'='*60}\n")
        f.write(output)
        f.write("\n")

    # Parse ratio from final table result line: "Table xxx result: ... ratio=X.XXx"
    # Use findall to get all matches, then take the last one (final result)
    ratio_matches = re.findall(r'ratio=(\d+\.\d+)x', output)
    ratio = float(ratio_matches[-1]) if ratio_matches else 0.0

    # Check if import failed
    if "IMPORT FAILED" in output or "import error" in output.lower():
        return ValidationResult(
            table_name=table_name,
            passed=False,
            gen_time=0,
            compare_time=0,
            ratio=ratio,
            error="Import failed"
        )

    # Determine pass/fail
    if is_drift_experiment:
        # For drift experiments: ratio must be >= threshold (lower bound)
        passed = ratio >= exp.threshold
        if passed:
            print(f"  Drift check PASSED: ratio={ratio:.2f}x >= {exp.threshold:.2f}")
        else:
            print(f"  Drift check FAILED: ratio={ratio:.2f}x < {exp.threshold:.2f}")
    else:
        # For ref experiments: use vd command's result
        passed = "PASSED" in output or "SUCCESS" in output

    return ValidationResult(
        table_name=table_name,
        passed=passed,
        gen_time=0,
        compare_time=0,
        ratio=ratio,
        error=None if passed else f"ratio={ratio:.2f}x not meeting threshold"
    )


def validate_experiment(
    exp: ExperimentConfig,
    baseline_perf: Dict[str, float],
    validated_tables: set = None,
) -> Dict[str, ValidationResult]:
    """Validate all tables for an experiment."""

    if validated_tables is None:
        validated_tables = set()

    results = {}

    print(f"\n{'#'*60}")
    print(f"EXPERIMENT: {exp.name}")
    print(f"{'#'*60}")
    print(f"  expdir: {exp.expdir_path}")
    print(f"  gen_db: {exp.gen_db_name}")
    print(f"  compare_db: {exp.compare_db}")
    print(f"  threshold: {exp.threshold}")

    # Check if expdir exists
    if not os.path.exists(exp.expdir_path):
        print(f"  Skipping: expdir not found")
        return results

    # Check which tables have generated data
    tables_with_data = []
    for table in DRIFTABLE_TABLES:
        if table_has_generated_data(exp.expdir_path, table):
            tables_with_data.append(table)

    if not tables_with_data:
        print(f"  Skipping: no generated data found")
        return results

    print(f"  Tables with data: {tables_with_data}")
    print(f"  Already validated: {list(validated_tables)}")

    # Filter tables that need validation
    tables_to_validate = [t for t in tables_with_data if t not in validated_tables]
    if not tables_to_validate:
        print(f"  All tables already validated")
        return results

    print(f"  To validate: {tables_to_validate}")

    # Ensure gen_db exists
    if not database_exists(exp.gen_db_name):
        print(f"  Creating database {exp.gen_db_name}...")
        if not create_database_from(exp.compare_db, exp.gen_db_name):
            print(f"  Failed to create database")
            return results

    # Validate each table
    is_drift_exp = exp.drift_value is not None

    for table in tables_to_validate:
        result = validate_table(exp, table, baseline_perf)
        results[table] = result

        if result.passed:
            print(f"  ✓ {table}: PASSED (ratio={result.ratio:.2f}x)")
            validated_tables.add(table)

            # Early stop for drift experiments: one table passed is enough
            if is_drift_exp:
                print(f"  [EARLY STOP] Drift experiment passed with {table}, skipping remaining tables")
                break
        else:
            print(f"  ✗ {table}: FAILED - {result.error}")
            # Continue to next table

    return results


# ============================================================================
# TASK 1: VALIDATE ALL EXPERIMENTS
# ============================================================================

def task1_validate_all():
    """Task 1: Validate all experiments."""
    print("\n" + "="*80)
    print("TASK 1: VALIDATE ALL EXPERIMENTS")
    print("="*80)

    # Get baseline performance
    os.makedirs("validation_logs", exist_ok=True)

    print("\n--- Getting baseline performance ---")
    imdb_ori_perf = get_baseline_performance(IMDB, "validation_logs/imdb_baseline.json")
    imdb_17_perf = get_baseline_performance(IMDB_2017, "validation_logs/imdb_2017_baseline.json")

    print(f"\nIMDB ORI total time: {sum(imdb_ori_perf.values()):.1f}ms ({len(imdb_ori_perf)} queries)")
    print(f"IMDB 17 total time: {sum(imdb_17_perf.values()):.1f}ms ({len(imdb_17_perf)} queries)")

    # Get all experiments
    experiments = get_all_experiments()
    print(f"\nTotal experiments: {len(experiments)}")

    # Track results
    all_results = {}
    validated_by_exp = {}

    # Load previous validation state if exists
    state_file = "validation_logs/validation_state.json"
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
            validated_by_exp = {k: set(v) for k, v in state.get("validated", {}).items()}
        print(f"Loaded previous state: {len(validated_by_exp)} experiments")

    # Validate each experiment
    for exp in experiments:
        # Get appropriate baseline
        if exp.reference_dataset:
            baseline_perf = imdb_17_perf  # Use imdb_17 for ref comparisons
        else:
            baseline_perf = imdb_ori_perf  # Use imdb_ori for drift comparisons

        # Get previously validated tables
        validated = validated_by_exp.get(exp.name, set())

        # Validate
        results = validate_experiment(exp, baseline_perf, validated)
        all_results[exp.name] = results
        validated_by_exp[exp.name] = validated

        # Save state after each experiment
        state = {
            "validated": {k: list(v) for k, v in validated_by_exp.items()},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    for exp_name, results in all_results.items():
        if results:
            passed = sum(1 for r in results.values() if r.passed)
            total = len(results)
            print(f"  {exp_name}: {passed}/{total} passed")

    return all_results, validated_by_exp


# ============================================================================
# TASK 2: TRAIN AND TEST BAO
# ============================================================================

# Path to schedule_bao_simple.sh relative to project root
SCHEDULE_BAO_SCRIPT = "benchmarks/lqos/bao/schedule_bao_simple.sh"


def task2_train_bao(validated_experiments: Dict[str, set]):
    """Task 2: Train and test BAO on validated data using schedule_bao_simple.sh."""
    print("\n" + "="*80)
    print("TASK 2: TRAIN AND TEST BAO")
    print("="*80)

    # Find experiments with all tables validated
    ready_experiments = []
    for exp_name, validated_tables in validated_experiments.items():
        if len(validated_tables) >= len(DRIFTABLE_TABLES):
            ready_experiments.append(exp_name)

    if not ready_experiments:
        print("No experiments with all tables validated. Skipping BAO training.")
        return

    print(f"Experiments ready for BAO training: {ready_experiments}")

    # Check if schedule_bao_simple.sh exists
    if not os.path.exists(SCHEDULE_BAO_SCRIPT):
        print(f"ERROR: {SCHEDULE_BAO_SCRIPT} not found")
        return

    # Get experiment configs
    all_experiments = {exp.name: exp for exp in get_all_experiments()}

    bao_results = {}
    for exp_name in ready_experiments:
        exp = all_experiments.get(exp_name)
        if not exp:
            continue

        print(f"\n{'#'*60}")
        print(f"Training and Testing BAO on: {exp_name}")
        print(f"  Database: {exp.gen_db_name}")
        print(f"{'#'*60}")

        # Check if database exists
        if not database_exists(exp.gen_db_name):
            print(f"  Skipping: database {exp.gen_db_name} does not exist")
            continue

        # Use schedule_bao_simple.sh for training and testing
        # Train on imdb (baseline), test on generated database
        # This tests how well BAO trained on original data performs on drifted data
        cmd = f"./{SCHEDULE_BAO_SCRIPT} " \
              f"--train-dataset imdb " \
              f"--test-dataset {exp.gen_db_name} " \
              f"--force-retest"  # Force retest to get fresh results

        success, output = run_command(
            cmd,
            f"Run schedule_bao_simple.sh for {exp_name}",
            capture=False  # Show output in real-time
        )

        if success:
            print(f"\n✓ BAO training and testing complete for {exp_name}")
            bao_results[exp_name] = "success"
        else:
            print(f"\n✗ BAO training/testing failed for {exp_name}")
            bao_results[exp_name] = "failed"

    # Print summary
    print("\n" + "="*80)
    print("BAO TRAINING/TESTING SUMMARY")
    print("="*80)
    for exp_name, status in bao_results.items():
        status_icon = "✓" if status == "success" else "✗"
        print(f"  {status_icon} {exp_name}: {status}")

    return bao_results


# ============================================================================
# MAIN
# ============================================================================

def validate_single(exp_name: str, table_name: str):
    """Quick test: validate a single experiment + table."""
    print(f"\n{'='*60}")
    print(f"QUICK TEST: {exp_name} / {table_name}")
    print(f"{'='*60}")

    # Find experiment
    all_exps = {e.name: e for e in get_all_experiments()}
    if exp_name not in all_exps:
        print(f"Error: experiment '{exp_name}' not found")
        print(f"Available: {list(all_exps.keys())}")
        return

    exp = all_exps[exp_name]
    print(f"  expdir: {exp.expdir_path}")
    print(f"  gen_db: {exp.gen_db_name}")
    print(f"  compare_db: {exp.compare_db}")
    print(f"  threshold: {exp.threshold}")

    # Check CSV exists
    csv_path = os.path.join(exp.expdir_path, table_name, f"{table_name}.drifted.csv")
    if not os.path.exists(csv_path):
        print(f"\nError: CSV not found: {csv_path}")
        return

    print(f"  csv: {csv_path} ✓")

    # Ensure gen_db exists
    if not database_exists(exp.gen_db_name):
        print(f"\nCreating database {exp.gen_db_name} from {exp.compare_db}...")
        if not create_database_from(exp.compare_db, exp.gen_db_name):
            print("Failed to create database")
            return

    # Validate
    result = validate_table(exp, table_name, {})
    print(f"\nResult: {'PASSED' if result.passed else 'FAILED'}")
    print(f"  ratio: {result.ratio:.2f}x")
    if result.error:
        print(f"  error: {result.error}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate experiments and train BAO")
    parser.add_argument("--task", type=int, choices=[1, 2], default=None,
                        help="Task to run (1=validate, 2=train BAO, default=both)")
    parser.add_argument("--exp", type=str, default=None,
                        help="Specific experiment (e.g., 'task1_imdb_to_imdb_2017')")
    parser.add_argument("--table", type=str, default=None,
                        help="Specific table to validate (e.g., 'aka_name')")
    parser.add_argument("--list", action="store_true",
                        help="List all experiments and exit")
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for exp in get_all_experiments():
            print(f"  {exp.name}")
            print(f"    expdir: {exp.expdir_path}")
            print(f"    gen_db: {exp.gen_db_name}")
            print(f"    compare_db: {exp.compare_db}")
        return

    # Quick test mode: single exp + table
    if args.exp and args.table:
        validate_single(args.exp, args.table)
        return

    # Setup logging - save console output to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"validation_logs/{timestamp}"
    log_file = f"{log_dir}/validation.log"
    logger = TeeLogger(log_file)
    sys.stdout = logger
    print(f"Validation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")

    start_time = time.time()

    validated_experiments = {}

    try:
        if args.task is None or args.task == 1:
            _, validated_experiments = task1_validate_all()

        if args.task is None or args.task == 2:
            # Load state if we didn't run task 1
            if not validated_experiments:
                state_file = "validation_logs/validation_state.json"
                if os.path.exists(state_file):
                    with open(state_file) as f:
                        state = json.load(f)
                        validated_experiments = {k: set(v) for k, v in state.get("validated", {}).items()}

            task2_train_bao(validated_experiments)

        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTotal time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
