#!/bin/bash
#
# LQO Baseline Training and Testing Script
# Train BAO, Balsa, Lero on IMDB and compare with PostgreSQL
#
# Usage: ./scripts/run_lqo_baseline.sh [OPTIONS]
#
# Options:
#   --dataset DATASET           Dataset to use (default: imdb)
#   --query-set QUERY_SET       Query set to use (default: job)
#   --systems SYSTEMS           Comma-separated systems to run (default: bao,balsa,lero)
#   --force-retrain             Force retraining even if model exists
#   --force-retest              Force retesting even if results exist
#   --batch-test-datasets LIST  Comma-separated datasets for batch testing
#   --train-dataset DATASET     Dataset used for training (for batch test mode, model lookup)
#   --parallel                  Run systems in parallel (experimental)
#   -h, --help                  Show this help message
#

set -e

# Disable Python output buffering for real-time logs
export PYTHONUNBUFFERED=1

# Activate conda environment (for docker exec -c mode)
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate ai4db_new
    export PATH="/root/miniconda3/envs/ai4db_new/bin:$PATH"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect script location and find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Default configuration
DATASET="imdb"
QUERY_SET="job"
SYSTEMS="bao,balsa,lero"
FORCE_RETRAIN=""
FORCE_RETEST=""
PARALLEL=false
BATCH_TEST_DATASETS=""
TRAIN_DATASET=""
MIN_QUERIES=""
CONTINUE_TRAINING=""
TEST_INTERVAL=""
CHECKPOINT_INTERVAL=""

# Parse arguments
show_help() {
    echo "LQO Baseline Training and Testing Script"
    echo ""
    echo "Train BAO, Balsa, Lero on a dataset and compare with PostgreSQL"
    echo ""
    echo "Usage: ./scripts/run_lqo_baseline.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dataset DATASET      Dataset to use (default: imdb)"
    echo "  --query-set QUERY_SET  Query set to use (default: job)"
    echo "  --systems SYSTEMS      Comma-separated systems: bao,balsa,lero (default: all)"
    echo "  --force-retrain        Force retraining even if model exists"
    echo "  --force-retest         Force retesting even if results exist"
    echo "  --min-queries N        Minimum training queries (sample with replacement)"
    echo "  --test-interval N      Run test every N iterations (Bao: default 20, Lero: default 25)"
    echo "  --checkpoint-interval N  Save checkpoint every N iterations (Bao only, default: 20)"
    echo "  --continue-training    Continue from existing model and data (Lero, Balsa)"
    echo "  --parallel             Run systems in parallel (experimental)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run all systems on imdb"
    echo "  ./scripts/run_lqo_baseline.sh"
    echo ""
    echo "  # Run only BAO and Lero"
    echo "  ./scripts/run_lqo_baseline.sh --systems bao,lero"
    echo ""
    echo "  # Force retrain and retest"
    echo "  ./scripts/run_lqo_baseline.sh --force-retrain --force-retest"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --query-set)
            QUERY_SET="$2"
            shift 2
            ;;
        --systems)
            SYSTEMS="$2"
            shift 2
            ;;
        --force-retrain)
            FORCE_RETRAIN="--force-retrain"
            shift
            ;;
        --force-retest)
            FORCE_RETEST="--force-retest"
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --batch-test-datasets)
            BATCH_TEST_DATASETS="$2"
            shift 2
            ;;
        --train-dataset)
            TRAIN_DATASET="$2"
            shift 2
            ;;
        --min-queries)
            MIN_QUERIES="$2"
            shift 2
            ;;
        --continue-training)
            CONTINUE_TRAINING="--continue-training"
            shift
            ;;
        --test-interval)
            TEST_INTERVAL="$2"
            shift 2
            ;;
        --checkpoint-interval)
            CHECKPOINT_INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If batch test mode, set train dataset default
if [ -n "$BATCH_TEST_DATASETS" ] && [ -z "$TRAIN_DATASET" ]; then
    TRAIN_DATASET="$DATASET"
fi

# Configuration
LOG_DIR="./lqo_baseline_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${LOG_DIR}/${TIMESTAMP}_summary.txt"

mkdir -p "$LOG_DIR"

echo "================================================================"
echo -e "${BLUE}LQO Baseline Training and Testing${NC}"
echo "================================================================"
echo "Dataset:   $DATASET"
echo "Query Set: $QUERY_SET"
echo "Systems:   $SYSTEMS"
echo "Start:     $(date)"
echo "================================================================"
echo ""

# Record start time
START_TIME=$(date +%s)

# Initialize summary
cat > "$SUMMARY_FILE" << EOF
LQO Baseline Results
====================
Dataset:   $DATASET
Query Set: $QUERY_SET
Date:      $(date)

EOF

# Function to run a system
run_system() {
    local system="$1"
    local script=""
    local log_file="${LOG_DIR}/${TIMESTAMP}_${system}.log"

    case $system in
        bao)
            script="benchmarks/lqos/bao/schedule_bao_simple.sh"
            ;;
        balsa)
            script="benchmarks/lqos/balsa/schedule_balsa_simple.sh"
            ;;
        lero)
            script="benchmarks/lqos/Lero/schedule_lero_simple.sh"
            ;;
        *)
            echo -e "${RED}Unknown system: $system${NC}"
            return 1
            ;;
    esac

    if [ ! -f "$script" ]; then
        echo -e "${RED}Script not found: $script${NC}"
        return 1
    fi

    echo -e "${BLUE}Running $system...${NC}"
    echo "Log: $log_file"

    local train_ds="${TRAIN_DATASET:-$DATASET}"
    local test_ds="$DATASET"

    # Build min-queries argument if specified
    local min_queries_arg=""
    if [ -n "$MIN_QUERIES" ]; then
        min_queries_arg="--min-queries $MIN_QUERIES"
    fi

    # Build continue-training argument if specified (Lero and Balsa)
    local continue_training_arg=""
    if [ -n "$CONTINUE_TRAINING" ] && [[ "$system" == "lero" || "$system" == "balsa" ]]; then
        continue_training_arg="$CONTINUE_TRAINING"
    fi

    # Build system-specific arguments
    local system_args=""
    if [[ "$system" == "bao" ]]; then
        if [ -n "$TEST_INTERVAL" ]; then
            system_args="$system_args --test-interval $TEST_INTERVAL"
        fi
        if [ -n "$CHECKPOINT_INTERVAL" ]; then
            system_args="$system_args --checkpoint-interval $CHECKPOINT_INTERVAL"
        fi
    elif [[ "$system" == "lero" ]]; then
        if [ -n "$TEST_INTERVAL" ]; then
            system_args="$system_args --test-interval $TEST_INTERVAL"
        fi
    fi

    # Note: BAO/Balsa/Lero scripts auto-skip training if model exists
    local cmd="./$script --train-dataset $train_ds --train-query-set $QUERY_SET --test-dataset $test_ds --test-query-set $QUERY_SET $FORCE_RETRAIN $FORCE_RETEST $min_queries_arg $continue_training_arg $system_args"

    echo "Command: $cmd"
    echo ""

    local sys_start=$(date +%s)

    if $cmd 2>&1 | tee "$log_file"; then
        local sys_end=$(date +%s)
        local sys_time=$((sys_end - sys_start))
        echo -e "${GREEN}✓ $system completed in ${sys_time}s${NC}"
        echo "$system: SUCCESS (${sys_time}s)" >> "$SUMMARY_FILE"
        return 0
    else
        local sys_end=$(date +%s)
        local sys_time=$((sys_end - sys_start))
        echo -e "${RED}✗ $system failed after ${sys_time}s${NC}"
        echo "$system: FAILED (${sys_time}s)" >> "$SUMMARY_FILE"
        return 1
    fi
}

# ============================================================
# Batch Test Mode: Test on multiple datasets
# ============================================================
if [ -n "$BATCH_TEST_DATASETS" ]; then
    echo -e "${YELLOW}Batch Test Mode${NC}"
    echo "Train Dataset: $TRAIN_DATASET"
    echo "Test Datasets: $BATCH_TEST_DATASETS"
    echo ""

    IFS=',' read -ra TEST_DATASETS <<< "$BATCH_TEST_DATASETS"
    BATCH_RESULTS=()

    for test_ds in "${TEST_DATASETS[@]}"; do
        test_ds=$(echo "$test_ds" | tr -d ' ')  # trim whitespace
        echo ""
        echo "================================================================"
        echo -e "${YELLOW}Testing on dataset: $test_ds${NC}"
        echo "================================================================"

        # Run each system on this test dataset
        IFS=',' read -ra SYSTEM_ARRAY <<< "$SYSTEMS"
        for system in "${SYSTEM_ARRAY[@]}"; do
            system=$(echo "$system" | tr -d ' ')
            echo -e "${BLUE}[$system] Testing on $test_ds...${NC}"

            case $system in
                bao)
                    script="benchmarks/lqos/bao/schedule_bao_simple.sh"
                    ;;
                balsa)
                    script="benchmarks/lqos/balsa/schedule_balsa_simple.sh"
                    ;;
                lero)
                    script="benchmarks/lqos/Lero/schedule_lero_simple.sh"
                    ;;
                *)
                    echo -e "${RED}Unknown system: $system${NC}"
                    continue
                    ;;
            esac

            log_file="${LOG_DIR}/${TIMESTAMP}_${system}_${test_ds}.log"
            cmd="./$script --train-dataset $TRAIN_DATASET --train-query-set $QUERY_SET --test-dataset $test_ds --test-query-set $QUERY_SET --force-retest"

            echo "Command: $cmd"
            if $cmd 2>&1 | tee "$log_file"; then
                echo -e "${GREEN}✓ $system on $test_ds completed${NC}"
                BATCH_RESULTS+=("$system:$test_ds:SUCCESS")
            else
                echo -e "${RED}✗ $system on $test_ds failed${NC}"
                BATCH_RESULTS+=("$system:$test_ds:FAILED")
            fi
        done
    done

    # Print batch results summary
    echo ""
    echo "================================================================"
    echo -e "${GREEN}Batch Test Complete!${NC}"
    echo "================================================================"
    echo "Results:"
    for result in "${BATCH_RESULTS[@]}"; do
        echo "  $result"
    done
    echo ""
    echo "Logs: $LOG_DIR/"
    exit 0
fi

# ============================================================
# Normal Mode: Train and test on single dataset
# ============================================================

# Run each system
IFS=',' read -ra SYSTEM_ARRAY <<< "$SYSTEMS"

FAILED_SYSTEMS=()
SUCCESS_SYSTEMS=()

for system in "${SYSTEM_ARRAY[@]}"; do
    system=$(echo "$system" | tr -d ' ')  # trim whitespace
    echo ""
    echo "================================================================"
    echo -e "${YELLOW}[$system] Starting...${NC}"
    echo "================================================================"

    if run_system "$system"; then
        SUCCESS_SYSTEMS+=("$system")
    else
        FAILED_SYSTEMS+=("$system")
    fi

    echo ""
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

# Generate comparison report
echo "" >> "$SUMMARY_FILE"
echo "Performance Comparison" >> "$SUMMARY_FILE"
echo "======================" >> "$SUMMARY_FILE"

# Try to extract latencies from logs
echo ""
echo "================================================================"
echo -e "${BLUE}Extracting Performance Results...${NC}"
echo "================================================================"

python3 << EOF
import os
import re
import glob

log_dir = "$LOG_DIR"
timestamp = "$TIMESTAMP"
dataset = "$DATASET"
query_set = "$QUERY_SET"

results = {}

def extract_total_time_from_log(log_file):
    """Extract total time from log file with pattern 'total time: XXXms'"""
    if not os.path.exists(log_file):
        return None
    with open(log_file) as f:
        content = f.read()
    # Match: "total time: 164439.0ms" or "total time: 164439ms"
    match = re.search(r'total time:\s*(\d+\.?\d*)\s*ms', content, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def extract_total_from_txt(txt_file):
    """Extract total execution time from CSV-like txt file (6th column is execution time)"""
    if not os.path.exists(txt_file):
        return None
    total = 0.0
    with open(txt_file) as f:
        for line in f:
            parts = line.strip().split(', ')
            if len(parts) >= 6:
                try:
                    exec_time = float(parts[5])  # 6th column (0-indexed: 5)
                    total += exec_time
                except (ValueError, IndexError):
                    continue
    return total if total > 0 else None

# BAO results - try .log first, then .txt
bao_log = sorted(glob.glob(f"bao_logs_all/*_test_bao_{dataset}_{query_set}.log"))
pg_log = sorted(glob.glob(f"bao_logs_all/*_test_pg_{dataset}_{query_set}.log"))
bao_txt = sorted(glob.glob(f"bao_logs_all/test_bao_{query_set}_*.txt"))
pg_txt = sorted(glob.glob(f"bao_logs_all/test_pg_{query_set}_*.txt"))

# Get most recent files
if bao_log:
    total = extract_total_time_from_log(bao_log[-1])
    if total:
        results['BAO'] = total
if not results.get('BAO') and bao_txt:
    total = extract_total_from_txt(bao_txt[-1])
    if total:
        results['BAO'] = total

if pg_log:
    total = extract_total_time_from_log(pg_log[-1])
    if total:
        results['PostgreSQL'] = total
if not results.get('PostgreSQL') and pg_txt:
    total = extract_total_from_txt(pg_txt[-1])
    if total:
        results['PostgreSQL'] = total

# Balsa results - parse "Execution time: XXXX.X" format
balsa_log = sorted(glob.glob(f"balsa_logs_all/*_test_balsa_{dataset}_{query_set}.log"))
if balsa_log:
    log_file = balsa_log[-1]
    if os.path.exists(log_file):
        with open(log_file) as f:
            content = f.read()
        # Match: "Execution time: 5743.6 (predicted"
        exec_times = re.findall(r'Execution time:\s*(\d+\.?\d*)', content)
        if exec_times:
            results['Balsa'] = sum(float(t) for t in exec_times)

# Lero results - parse "after writting write_latency_file <timestamp> <query_id> <latency_ms>" format
def extract_lero_latency(log_file):
    """Extract total latency from Lero log file by summing all query latencies"""
    if not os.path.exists(log_file):
        return None
    total = 0.0
    count = 0
    with open(log_file) as f:
        for line in f:
            # Match: "after writting write_latency_file 1768472431.3157835 10a 369.305"
            match = re.search(r'after writting write_latency_file\s+[\d.]+\s+\S+\s+([\d.]+)', line)
            if match:
                total += float(match.group(1))
                count += 1
    return total if count > 0 else None

# Try lero_logs_all first (test output logs)
lero_output_log = sorted(glob.glob(f"lero_logs_all/*_test_lero_output_{dataset}_{query_set}.log"))
if lero_output_log:
    total = extract_lero_latency(lero_output_log[-1])
    if total:
        results['Lero'] = total

# If not found, try lqo_baseline_logs (run_lqo_baseline logs)
if not results.get('Lero'):
    lero_baseline_log = sorted(glob.glob(f"{log_dir}/*_lero.log"))
    if lero_baseline_log:
        total = extract_lero_latency(lero_baseline_log[-1])
        if total:
            results['Lero'] = total

# Fallback: try the standard log format
if not results.get('Lero'):
    lero_log = sorted(glob.glob(f"lero_logs_all/*_test_lero_{dataset}_{query_set}.log"))
    if lero_log:
        total = extract_total_time_from_log(lero_log[-1])
        if total:
            results['Lero'] = total

# Print results
if results:
    print("\nPerformance Summary (Total Query Latency):")
    print("-" * 40)

    pg_time = results.get('PostgreSQL', 0)

    for system in ['PostgreSQL', 'BAO', 'Balsa', 'Lero']:
        if system in results:
            latency = results[system]
            if pg_time > 0 and system != 'PostgreSQL':
                speedup = pg_time / latency
                print(f"  {system:12}: {latency:10.1f} ms  ({speedup:.2f}x vs PG)")
            else:
                print(f"  {system:12}: {latency:10.1f} ms")

    # Save to summary
    with open("$SUMMARY_FILE", 'a') as f:
        f.write("\nTotal Query Latency (ms):\n")
        for system in ['PostgreSQL', 'BAO', 'Balsa', 'Lero']:
            if system in results:
                latency = results[system]
                if pg_time > 0 and system != 'PostgreSQL':
                    speedup = pg_time / latency
                    f.write(f"  {system:12}: {latency:10.1f} ms  ({speedup:.2f}x vs PG)\n")
                else:
                    f.write(f"  {system:12}: {latency:10.1f} ms\n")
else:
    print("\nNo performance results found in logs.")
    print("Check individual log files for details.")
EOF

# Final summary
echo ""
echo "================================================================"
echo -e "${GREEN}Pipeline Complete!${NC}"
echo "================================================================"
echo "Total Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Successful: ${SUCCESS_SYSTEMS[*]:-none}"
echo "Failed:     ${FAILED_SYSTEMS[*]:-none}"
echo ""
echo "Summary:    $SUMMARY_FILE"
echo "Logs:       $LOG_DIR/"
echo ""
echo "Individual logs:"
echo "  BAO:   bao_logs_all/"
echo "  Balsa: balsa_logs_all/"
echo "  Lero:  lero_logs_all/"
echo ""
echo "End Time: $(date)"
echo "================================================================"

# Append timing to summary
echo "" >> "$SUMMARY_FILE"
echo "Timing" >> "$SUMMARY_FILE"
echo "======" >> "$SUMMARY_FILE"
echo "Total Time: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "$SUMMARY_FILE"
echo "Successful: ${SUCCESS_SYSTEMS[*]:-none}" >> "$SUMMARY_FILE"
echo "Failed: ${FAILED_SYSTEMS[*]:-none}" >> "$SUMMARY_FILE"

# Exit with error if any system failed
if [ ${#FAILED_SYSTEMS[@]} -gt 0 ]; then
    exit 1
fi
