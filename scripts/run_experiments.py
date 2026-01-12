#!/usr/bin/env python3
"""
Experiment runner script for drift tuning and timing tests.

Task 1: Tune drift ref (base datasets -> imdb_2017)
Task 2: Tune drift val (imdb with different drift values)
Task 3: Generate 2015 data using imdb, imdb_2013, imdb_2014 as training data
Task 4: Train time + gen time measurement
"""

import os
import sys
import subprocess
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import threading


class TeeLogger:
    """Write output to both console and log file."""
    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8', buffering=1)  # Line buffered

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Which tasks to run
RUN_TASK_1 = False
RUN_TASK_2 = False
RUN_TASK_3 = False
RUN_TASK_4 = True

# GPU allocation (all tasks use 2 GPUs per job)
GPUS_PER_JOB = 2

# All available GPUs (adjust based on your system)
# ALL_GPUS = [2, 3, 4, 5, 6, 7]
ALL_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

# Task 3: Generate 2015 using historical data
TASK_3_BASES = ["imdb", "imdb_2013", "imdb_2014"]  # Training data
TASK_3_TARGET = "imdb_2015"  # Target to generate

# Task 4 configurations:
# Part 1: By dataset size (fixed GPU count)
TASK_4_PART1_GPU = 4  # Use 4 GPUs
TASK_4_PART1_REFS = ["imdb_2013", "imdb_2015", "imdb_2017"]  # Datasets to compare

# Part 2: By GPU count (fixed dataset)
TASK_4_PART2_GPUS = [8]  # GPU counts to compare (only 8 for now)
TASK_4_PART2_REF = "imdb_2013"  # Fixed dataset

# Which parts to run
RUN_TASK_4_PART1 = False  # By dataset size
RUN_TASK_4_PART2 = True  # By GPU count

# Sleep time after each job completes (seconds)
JOB_COMPLETE_SLEEP = 10

# ============================================================================
# TASK DEFINITIONS
# ============================================================================

# Task 1: Base datasets for drift ref tuning
TASK_1_BASES = ["imdb", "imdb_2013", "imdb_2014", "imdb_2015", "imdb_2016"]
TASK_1_REF = "imdb_2017"
DRIFT_REF_FILE = "drift_ref.csv"

# Task 2: Drift values to test
TASK_2_DRIFT_VALUES = [0.1, 0.3, 0.5]

# Task 4: Base dataset
TASK_4_BASE = "imdb"

# ============================================================================
# GPU ALLOCATION MANAGER
# ============================================================================

class GPUAllocator:
    """Thread-safe GPU allocator."""

    def __init__(self, gpus: List[int]):
        self.available_gpus = list(gpus)
        self.lock = threading.Lock()

    def allocate(self, count: int) -> Optional[List[int]]:
        """Allocate `count` GPUs. Returns None if not enough available."""
        with self.lock:
            if len(self.available_gpus) >= count:
                allocated = self.available_gpus[:count]
                self.available_gpus = self.available_gpus[count:]
                return allocated
            return None

    def release(self, gpus: List[int]):
        """Release GPUs back to the pool."""
        with self.lock:
            self.available_gpus.extend(gpus)

    def wait_and_allocate(self, count: int, poll_interval: float = 1.0) -> List[int]:
        """Wait until `count` GPUs are available, then allocate."""
        while True:
            result = self.allocate(count)
            if result is not None:
                return result
            time.sleep(poll_interval)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def run_command(cmd: str, description: str = "", log_file: str = None) -> Tuple[bool, float]:
    """Run a command with real-time log output and return (success, elapsed_time)."""
    print(f"\n{'='*60}", flush=True)
    print(f"Running: {description or cmd}", flush=True)
    print(f"{'='*60}", flush=True)

    start_time = time.time()

    try:
        if log_file:
            # Use Popen for real-time output to both console and log file
            with open(log_file, "w", buffering=1) as f:  # Line buffered
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1  # Line buffered
                )

                # Read and write output line by line
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                    print(line, end='', flush=True)

                process.wait()
                returncode = process.returncode
        else:
            result = subprocess.run(cmd, shell=True)
            returncode = result.returncode

        elapsed = time.time() - start_time
        success = returncode == 0

        status = "SUCCESS" if success else "FAILED"
        print(f"{status} in {elapsed:.1f}s", flush=True)

        return success, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"ERROR: {e}", flush=True)
        return False, elapsed


def check_drift_ref_exists(base: str, ref: str, drift_ref_file: str) -> bool:
    """Check if drift ref entry exists for base->ref in the CSV file."""
    if not os.path.exists(drift_ref_file):
        return False

    try:
        import pandas as pd
        df = pd.read_csv(drift_ref_file)
        # Check if there's an entry with matching src_dataset and dst_dataset
        mask = (df['src_dataset'] == base) & (df['dst_dataset'] == ref)
        return mask.any()
    except Exception as e:
        print(f"Warning: Error reading {drift_ref_file}: {e}", flush=True)
        return False


def format_gpus(gpus: List[int]) -> str:
    """Format GPU list as comma-separated string."""
    return ",".join(map(str, gpus))


# ============================================================================
# TASK 1: TUNE DRIFT REF
# ============================================================================

def exp_1_job(base: str, ref: str, gpus: List[int], log_dir: str) -> Dict:
    """Run a single Task 1 job (drift ref tuning for one base dataset)."""
    result = {
        "task": "task_1",
        "base": base,
        "ref": ref,
        "gpus": gpus,
        "success": False,
        "elapsed_time": 0,
    }

    gpu_str = format_gpus(gpus)
    log_file = os.path.join(log_dir, f"task1_{base}_to_{ref}.log")

    # Step 1: Check if drift ref exists, if not, calculate it
    if not check_drift_ref_exists(base, ref, DRIFT_REF_FILE):
        print(f"[Task 1] Calculating drift ref: {base} -> {ref}", flush=True)
        cmd = f"python calc_drift.py --src-dir datasets/{base} --dst-dir datasets/{ref} --output {DRIFT_REF_FILE}"
        success, _ = run_command(cmd, f"calc_drift: {base} -> {ref}")
        if not success:
            result["error"] = "calc_drift failed"
            return result
    else:
        print(f"[Task 1] Drift ref exists: {base} -> {ref}", flush=True)

    # Step 2: Run auto-tune with drift ref
    cmd = f"python cli.py gd {base} --drift-ref={DRIFT_REF_FILE} --ref={ref} --exclude=title --ops=all --sample-steps=200 --gpus={gpu_str}"

    success, elapsed = run_command(cmd, f"Task 1: {base} -> {ref} (GPUs: {gpu_str})", log_file)

    result["success"] = success
    result["elapsed_time"] = elapsed

    # Sleep after job completes
    print(f"[Task 1] Sleeping {JOB_COMPLETE_SLEEP}s after {base} -> {ref}...", flush=True)
    time.sleep(JOB_COMPLETE_SLEEP)

    return result


def exp_1(gpu_allocator: GPUAllocator, log_dir: str) -> List[Dict]:
    """Run all Task 1 jobs with GPU allocation."""
    print("\n" + "="*80, flush=True)
    print("TASK 1: TUNE DRIFT REF", flush=True)
    print("="*80, flush=True)

    results = []

    def run_job(base):
        gpus = gpu_allocator.wait_and_allocate(GPUS_PER_JOB)
        try:
            result = exp_1_job(base, TASK_1_REF, gpus, log_dir)
            return result
        finally:
            gpu_allocator.release(gpus)

    with ThreadPoolExecutor(max_workers=len(TASK_1_BASES)) as executor:
        futures = {executor.submit(run_job, base): base for base in TASK_1_BASES}

        for future in as_completed(futures):
            base = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[Task 1] Completed: {base} - {'SUCCESS' if result['success'] else 'FAILED'}", flush=True)
            except Exception as e:
                print(f"[Task 1] Error for {base}: {e}", flush=True)
                results.append({"task": "task_1", "base": base, "success": False, "error": str(e)})

    return results


# ============================================================================
# TASK 2: TUNE DRIFT VAL
# ============================================================================

def exp_2_job(drift_val: float, gpus: List[int], log_dir: str) -> Dict:
    """Run a single Task 2 job (drift val tuning)."""
    result = {
        "task": "task_2",
        "drift_val": drift_val,
        "gpus": gpus,
        "success": False,
        "elapsed_time": 0,
    }

    gpu_str = format_gpus(gpus)
    log_file = os.path.join(log_dir, f"task2_drift_{drift_val}.log")

    # Use drift value to create variant-id (0.1 -> 1, 0.3 -> 3, 0.5 -> 5)
    variant_id = int(drift_val * 10)
    cmd = f"python cli.py gd imdb {drift_val} --exclude=title --ops=all --sample-steps=200 --variant-id={variant_id} --gpus={gpu_str}"

    success, elapsed = run_command(cmd, f"Task 2: drift={drift_val} (GPUs: {gpu_str})", log_file)

    result["success"] = success
    result["elapsed_time"] = elapsed

    # Sleep after job completes
    print(f"[Task 2] Sleeping {JOB_COMPLETE_SLEEP}s after drift={drift_val}...", flush=True)
    time.sleep(JOB_COMPLETE_SLEEP)

    return result


def exp_2(gpu_allocator: GPUAllocator, log_dir: str) -> List[Dict]:
    """Run all Task 2 jobs with GPU allocation."""
    print("\n" + "="*80, flush=True)
    print("TASK 2: TUNE DRIFT VAL", flush=True)
    print("="*80, flush=True)

    results = []

    def run_job(drift_val):
        gpus = gpu_allocator.wait_and_allocate(GPUS_PER_JOB)
        try:
            result = exp_2_job(drift_val, gpus, log_dir)
            return result
        finally:
            gpu_allocator.release(gpus)

    with ThreadPoolExecutor(max_workers=len(TASK_2_DRIFT_VALUES)) as executor:
        futures = {executor.submit(run_job, dv): dv for dv in TASK_2_DRIFT_VALUES}

        for future in as_completed(futures):
            drift_val = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[Task 2] Completed: drift={drift_val} - {'SUCCESS' if result['success'] else 'FAILED'}", flush=True)
            except Exception as e:
                print(f"[Task 2] Error for drift={drift_val}: {e}", flush=True)
                results.append({"task": "task_2", "drift_val": drift_val, "success": False, "error": str(e)})

    return results


# ============================================================================
# TASK 3: GENERATE 2015 USING HISTORICAL DATA
# ============================================================================

def exp_3_job(base: str, target: str, gpus: List[int], log_dir: str) -> Dict:
    """Run a single Task 3 job (generate target using base as training data)."""
    result = {
        "task": "task_3",
        "base": base,
        "target": target,
        "gpus": gpus,
        "success": False,
        "elapsed_time": 0,
    }

    gpu_str = format_gpus(gpus)
    log_file = os.path.join(log_dir, f"task3_{base}_to_{target}.log")

    # Step 1: Check if drift ref exists, if not, calculate it
    if not check_drift_ref_exists(base, target, DRIFT_REF_FILE):
        print(f"[Task 3] Calculating drift ref: {base} -> {target}", flush=True)
        cmd = f"python calc_drift.py --src-dir datasets/{base} --dst-dir datasets/{target} --output {DRIFT_REF_FILE}"
        success, _ = run_command(cmd, f"calc_drift: {base} -> {target}")
        if not success:
            result["error"] = "calc_drift failed"
            return result
    else:
        print(f"[Task 3] Drift ref exists: {base} -> {target}", flush=True)

    # Step 2: Run generation with drift ref
    cmd = f"python cli.py gd {base} --drift-ref={DRIFT_REF_FILE} --ref={target} --exclude=title --ops=all --sample-steps=200 --gpus={gpu_str}"

    success, elapsed = run_command(cmd, f"Task 3: {base} -> {target} (GPUs: {gpu_str})", log_file)

    result["success"] = success
    result["elapsed_time"] = elapsed

    # Sleep after job completes
    print(f"[Task 3] Sleeping {JOB_COMPLETE_SLEEP}s after {base} -> {target}...", flush=True)
    time.sleep(JOB_COMPLETE_SLEEP)

    return result


def exp_3(gpu_allocator: GPUAllocator, log_dir: str) -> List[Dict]:
    """Run all Task 3 jobs (generate 2015 from historical data)."""
    print("\n" + "="*80, flush=True)
    print("TASK 3: GENERATE 2015 USING HISTORICAL DATA", flush=True)
    print("="*80, flush=True)

    results = []

    def run_job(base):
        gpus = gpu_allocator.wait_and_allocate(GPUS_PER_JOB)
        try:
            result = exp_3_job(base, TASK_3_TARGET, gpus, log_dir)
            return result
        finally:
            gpu_allocator.release(gpus)

    with ThreadPoolExecutor(max_workers=len(TASK_3_BASES)) as executor:
        futures = {executor.submit(run_job, base): base for base in TASK_3_BASES}

        for future in as_completed(futures):
            base = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[Task 3] Completed: {base} -> {TASK_3_TARGET} - {'SUCCESS' if result['success'] else 'FAILED'}", flush=True)
            except Exception as e:
                print(f"[Task 3] Error for {base}: {e}", flush=True)
                results.append({"task": "task_3", "base": base, "success": False, "error": str(e)})

    return results


# ============================================================================
# TASK 4: TRAIN TIME + GEN TIME
# ============================================================================

def exp_4_job(base: str, ref: str, gpus: List[int], log_dir: str) -> Dict:
    """Run a single Task 4 job (timing measurement).

    Runs two phases:
    1. ops=retrain-only: Train only, measure training time
    2. ops=regene-only: Generate only, measure generation time
    """
    num_gpus = len(gpus)
    result = {
        "task": "task_4",
        "base": base,
        "ref": ref,
        "num_gpus": num_gpus,
        "gpus": gpus,
        "success": False,
        "train_time": 0,
        "gen_time": 0,
        "total_time": 0,
    }

    gpu_str = format_gpus(gpus)
    variant_id = num_gpus  # Use num_gpus as variant ID to avoid overwriting

    # First, ensure drift ref exists
    if not check_drift_ref_exists(base, ref, DRIFT_REF_FILE):
        print(f"[Task 4] Calculating drift ref: {base} -> {ref}", flush=True)
        cmd = f"python calc_drift.py --src-dir datasets/{base} --dst-dir datasets/{ref} --output {DRIFT_REF_FILE}"
        success, _ = run_command(cmd, f"calc_drift: {base} -> {ref}")
        if not success:
            result["error"] = "calc_drift failed"
            return result

    # Phase 1: Train only (ops=retrain-only with variant-id)
    train_log_file = os.path.join(log_dir, f"task4_{base}_to_{ref}_{num_gpus}gpu_train.log")
    cmd_train = f"python cli.py gd {base} --drift-ref={DRIFT_REF_FILE} --ref={ref} --exclude=title --ops=retrain-only --variant-id={variant_id} --gpus={gpu_str}"

    train_success, train_elapsed = run_command(cmd_train, f"Task 4 Train: {base} -> {ref} ({num_gpus} GPUs)", train_log_file)
    result["train_time"] = train_elapsed

    if not train_success:
        result["error"] = "Training failed"
        return result

    # Phase 2: Generate only (ops=regene-only with variant-id)
    gen_log_file = os.path.join(log_dir, f"task4_{base}_to_{ref}_{num_gpus}gpu_gen.log")
    cmd_gen = f"python cli.py gd {base} --drift-ref={DRIFT_REF_FILE} --ref={ref} --exclude=title --ops=regene-only --variant-id={variant_id} --sample-steps=200 --gpus={gpu_str}"

    gen_success, gen_elapsed = run_command(cmd_gen, f"Task 4 Gen: {base} -> {ref} ({num_gpus} GPUs)", gen_log_file)
    result["gen_time"] = gen_elapsed

    result["success"] = gen_success
    result["total_time"] = train_elapsed + gen_elapsed

    # Sleep after job completes
    print(f"[Task 4] Sleeping {JOB_COMPLETE_SLEEP}s after {base} -> {ref} ({num_gpus} GPUs)...", flush=True)
    time.sleep(JOB_COMPLETE_SLEEP)

    return result


def exp_4(log_dir: str) -> List[Dict]:
    """Run all Task 4 jobs in two parts."""
    print("\n" + "="*80, flush=True)
    print("TASK 4: TRAIN TIME + GEN TIME", flush=True)
    print("="*80, flush=True)

    results = []

    # Part 1: By dataset size (fixed GPU count)
    if RUN_TASK_4_PART1:
        print(f"\n--- Part 1: By Dataset Size ({TASK_4_PART1_GPU} GPUs) ---", flush=True)
        print(f"    Datasets: {TASK_4_PART1_REFS}", flush=True)

        num_gpus = TASK_4_PART1_GPU
        max_concurrent = len(ALL_GPUS) // num_gpus
        print(f"    Max concurrent jobs: {max_concurrent}", flush=True)

        gpu_allocator = GPUAllocator(ALL_GPUS)

        def run_job_part1(ref):
            gpus = gpu_allocator.wait_and_allocate(num_gpus)
            try:
                result = exp_4_job(TASK_4_BASE, ref, gpus, log_dir)
                return result
            finally:
                gpu_allocator.release(gpus)

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {executor.submit(run_job_part1, ref): ref for ref in TASK_4_PART1_REFS}

            for future in as_completed(futures):
                ref = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "SUCCESS" if result["success"] else "FAILED"
                    print(f"[Task 4 Part1] {TASK_4_BASE} -> {ref} ({num_gpus} GPUs): {status}", flush=True)
                    print(f"              Train: {result['train_time']:.1f}s, Gen: {result['gen_time']:.1f}s, Total: {result['total_time']:.1f}s", flush=True)
                except Exception as e:
                    print(f"[Task 4 Part1] Error for {ref}: {e}", flush=True)
                    results.append({"task": "task_4", "ref": ref, "num_gpus": num_gpus, "success": False, "error": str(e)})

    # Part 2: By GPU count (fixed dataset)
    if RUN_TASK_4_PART2:
        print(f"\n--- Part 2: By GPU Count (Dataset: {TASK_4_PART2_REF}) ---", flush=True)
        print(f"    GPU configs: {TASK_4_PART2_GPUS}", flush=True)

        for num_gpus in TASK_4_PART2_GPUS:
            print(f"\n    Testing with {num_gpus} GPUs...", flush=True)

            gpu_allocator = GPUAllocator(ALL_GPUS)
            gpus = gpu_allocator.wait_and_allocate(num_gpus)

            try:
                result = exp_4_job(TASK_4_BASE, TASK_4_PART2_REF, gpus, log_dir)
                results.append(result)
                status = "SUCCESS" if result["success"] else "FAILED"
                print(f"[Task 4 Part2] {TASK_4_BASE} -> {TASK_4_PART2_REF} ({num_gpus} GPUs): {status}", flush=True)
                print(f"              Train: {result['train_time']:.1f}s, Gen: {result['gen_time']:.1f}s, Total: {result['total_time']:.1f}s", flush=True)
            except Exception as e:
                print(f"[Task 4 Part2] Error for {num_gpus} GPUs: {e}", flush=True)
                results.append({"task": "task_4", "ref": TASK_4_PART2_REF, "num_gpus": num_gpus, "success": False, "error": str(e)})
            finally:
                gpu_allocator.release(gpus)

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Create log directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"experiment_logs/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Setup TeeLogger to capture all console output
    master_log = f"{log_dir}/master.log"
    logger = TeeLogger(master_log)
    sys.stdout = logger

    print(f"Experiment started at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Master log: {master_log}", flush=True)
    print(f"Experiment logs will be saved to: {log_dir}", flush=True)
    print(f"\nConfiguration:", flush=True)
    print(f"  RUN_TASK_1: {RUN_TASK_1}", flush=True)
    print(f"  RUN_TASK_2: {RUN_TASK_2}", flush=True)
    print(f"  RUN_TASK_3: {RUN_TASK_3}", flush=True)
    print(f"  RUN_TASK_4: {RUN_TASK_4}", flush=True)
    print(f"  ALL_GPUS: {ALL_GPUS}", flush=True)
    print(f"  GPUS_PER_JOB: {GPUS_PER_JOB}", flush=True)
    print(f"  JOB_COMPLETE_SLEEP: {JOB_COMPLETE_SLEEP}s", flush=True)

    all_results = {
        "timestamp": timestamp,
        "config": {
            "run_task_1": RUN_TASK_1,
            "run_task_2": RUN_TASK_2,
            "run_task_3": RUN_TASK_3,
            "run_task_4": RUN_TASK_4,
            "all_gpus": ALL_GPUS,
        },
        "task_1": [],
        "task_2": [],
        "task_3": [],
        "task_4": [],
    }

    start_time = time.time()

    # Create shared GPU allocator for Task 1, 2, 3
    gpu_allocator = GPUAllocator(ALL_GPUS)

    # Run Task 1, 2, 3 in parallel using shared GPU pool
    print("\n" + "="*80, flush=True)
    print("RUNNING TASK 1, 2, 3 IN PARALLEL (shared GPU pool)", flush=True)
    print("="*80, flush=True)

    task_1_results = []
    task_2_results = []
    task_3_results = []

    def exp_1_wrapper():
        nonlocal task_1_results
        if RUN_TASK_1:
            task_1_results = exp_1(gpu_allocator, log_dir)

    def exp_2_wrapper():
        nonlocal task_2_results
        if RUN_TASK_2:
            task_2_results = exp_2(gpu_allocator, log_dir)

    def exp_3_wrapper():
        nonlocal task_3_results
        if RUN_TASK_3:
            task_3_results = exp_3(gpu_allocator, log_dir)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        if RUN_TASK_1:
            futures.append(executor.submit(exp_1_wrapper))
        if RUN_TASK_2:
            futures.append(executor.submit(exp_2_wrapper))
        if RUN_TASK_3:
            futures.append(executor.submit(exp_3_wrapper))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in parallel execution: {e}", flush=True)

    all_results["task_1"] = task_1_results
    all_results["task_2"] = task_2_results
    all_results["task_3"] = task_3_results

    # Run Task 4 sequentially (timing measurement)
    if RUN_TASK_4:
        task_4_results = exp_4(log_dir)
        all_results["task_4"] = task_4_results

    total_elapsed = time.time() - start_time
    all_results["total_elapsed_time"] = total_elapsed

    # Save results
    results_file = os.path.join(log_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*80, flush=True)
    print("EXPERIMENT SUMMARY", flush=True)
    print("="*80, flush=True)

    if RUN_TASK_1:
        print(f"\nTask 1 (Drift Ref Tuning):", flush=True)
        for r in all_results["task_1"]:
            status = "SUCCESS" if r.get("success") else "FAILED"
            print(f"  {r.get('base', 'N/A')} -> {r.get('ref', 'N/A')}: {status} ({r.get('elapsed_time', 0):.1f}s)", flush=True)

    if RUN_TASK_2:
        print(f"\nTask 2 (Drift Val Tuning):", flush=True)
        for r in all_results["task_2"]:
            status = "SUCCESS" if r.get("success") else "FAILED"
            print(f"  drift={r.get('drift_val', 'N/A')}: {status} ({r.get('elapsed_time', 0):.1f}s)", flush=True)

    if RUN_TASK_3:
        print(f"\nTask 3 (Generate 2015 from Historical Data):", flush=True)
        for r in all_results["task_3"]:
            status = "SUCCESS" if r.get("success") else "FAILED"
            print(f"  {r.get('base', 'N/A')} -> {r.get('target', 'N/A')}: {status} ({r.get('elapsed_time', 0):.1f}s)", flush=True)

    if RUN_TASK_4:
        print(f"\nTask 4 (Timing Measurement):", flush=True)
        for r in all_results["task_4"]:
            status = "SUCCESS" if r.get("success") else "FAILED"
            print(f"  {r.get('base', 'N/A')} -> {r.get('ref', 'N/A')} ({r.get('num_gpus', 'N/A')} GPUs): {status}", flush=True)
            print(f"    Train: {r.get('train_time', 0):.1f}s, Gen: {r.get('gen_time', 0):.1f}s, Total: {r.get('total_time', 0):.1f}s", flush=True)

    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal experiment time: {int(hours)}h {int(minutes)}m {seconds:.1f}s", flush=True)
    print(f"Results saved to: {results_file}", flush=True)

    # Close logger
    sys.stdout = logger.terminal
    logger.close()
    print(f"Master log saved to: {master_log}")


if __name__ == "__main__":
    main()
