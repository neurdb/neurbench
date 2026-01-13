#!/bin/bash
#
# LQO Baseline Training and Testing Script
# Train BAO, Balsa, Lero on IMDB and compare with PostgreSQL
#
# Usage: ./scripts/run_lqo_baseline.sh [OPTIONS]
#
# Options:
#   --dataset DATASET      Dataset to use (default: imdb)
#   --query-set QUERY_SET  Query set to use (default: job)
#   --systems SYSTEMS      Comma-separated systems to run (default: bao,balsa,lero)
#   --force-retrain        Force retraining even if model exists
#   --force-retest         Force retesting even if results exist
#   --parallel             Run systems in parallel (experimental)
#   -h, --help             Show this help message
#

set -e

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
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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

    local cmd="./$script --train-dataset $DATASET --train-query-set $QUERY_SET --test-dataset $DATASET --test-query-set $QUERY_SET $FORCE_RETRAIN $FORCE_RETEST"

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

# BAO results
bao_logs = glob.glob(f"bao_logs_all/*_test_bao_{dataset}_{query_set}.log") + \
           glob.glob(f"bao_logs_all/*_test_pg_{dataset}_{query_set}.log")

for log_file in bao_logs:
    if not os.path.exists(log_file):
        continue
    with open(log_file) as f:
        content = f.read()

    # Extract total latency
    # Look for patterns like "Total latency: XXX ms" or sum up individual query times
    latencies = re.findall(r'latency[:\s]+(\d+\.?\d*)\s*ms', content, re.IGNORECASE)
    if latencies:
        total = sum(float(l) for l in latencies)
        if 'test_bao' in log_file:
            results['BAO'] = total
        elif 'test_pg' in log_file:
            results['PostgreSQL'] = total

# Balsa results
balsa_logs = glob.glob(f"balsa_logs_all/*_test_balsa_{dataset}_{query_set}.log")
for log_file in balsa_logs:
    if not os.path.exists(log_file):
        continue
    with open(log_file) as f:
        content = f.read()
    latencies = re.findall(r'latency[:\s]+(\d+\.?\d*)\s*ms', content, re.IGNORECASE)
    if latencies:
        results['Balsa'] = sum(float(l) for l in latencies)

# Lero results
lero_logs = glob.glob(f"lero_logs_all/*_test_lero_{dataset}_{query_set}.log")
for log_file in lero_logs:
    if not os.path.exists(log_file):
        continue
    with open(log_file) as f:
        content = f.read()
    latencies = re.findall(r'latency[:\s]+(\d+\.?\d*)\s*ms', content, re.IGNORECASE)
    if latencies:
        results['Lero'] = sum(float(l) for l in latencies)

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
