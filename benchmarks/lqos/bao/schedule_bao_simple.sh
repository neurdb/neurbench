#!/bin/bash
#
# Bao Training and Testing Pipeline
# Usage: ./schedule_bao_simple.sh [QUERY_SET]
#
# Examples:
#   ./schedule_bao_simple.sh                           # Use default query_set
#   ./schedule_bao_simple.sh join-order-benchmark      # Use specified query_set
#

set -e  # Exit on error

# Detect script location and find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT"
echo "[INFO] Working directory: $(pwd)"

# Verify cli.py exists
if [ ! -f "cli.py" ]; then
    echo "ERROR: cli.py not found in project root: $PROJECT_ROOT"
    exit 1
fi

# Configuration
DATASET="imdb_ori"
QUERY_SET="${1:-join-order-benchmark}"  # Use first argument or default
LOG_DIR="./bao_logs_all"
BAO_SERVER_DIR="benchmarks/lqos/bao/bao_server"
TODAY=$(date +%Y%m%d)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================"
echo "Bao Training and Testing Pipeline (Clean Run)"
echo "Dataset: $DATASET"
echo "Query Set: $QUERY_SET"
echo "Start Time: $(date)"
echo "================================================================"

# Create log directory
mkdir -p $LOG_DIR

# Timing
START_TIME=$(date +%s)

# Step 1: Train Bao
echo ""
echo "[1/4] Training Bao with $DATASET..."
echo "----------------------------------------------------------------"
echo "[DEBUG] Query directory: queries/$QUERY_SET/train"
echo "[DEBUG] Database: $DATASET"
echo "[DEBUG] Port: 5430"
TRAIN_START=$(date +%s)

BAO_FORCE_CPU=1 python3 << EOF 2>&1 | tee $LOG_DIR/${TODAY}_train_${DATASET}_${QUERY_SET}.log
import sys
import os
os.environ['BAO_FORCE_CPU'] = '1'  # Force CPU to avoid CUDA multiprocessing issues
print("[DEBUG] Python script started")
sys.path.insert(0, '.')
print("[DEBUG] Importing cli module...")
from cli import GLOBAL_CONFIG, handle_tqo
print("[DEBUG] cli module imported")

GLOBAL_CONFIG['dataset'] = '$DATASET'
GLOBAL_CONFIG['query_set'] = '$QUERY_SET'
GLOBAL_CONFIG['pg_port'] = 5430
print(f"[DEBUG] Dataset: {GLOBAL_CONFIG['dataset']}")
print(f"[DEBUG] Query Set: {GLOBAL_CONFIG['query_set']}")
print(f"[DEBUG] PostgreSQL Port: {GLOBAL_CONFIG['pg_port']}")
print("[DEBUG] Executing: tqo bao")
sys.stdout.flush()
handle_tqo(['tqo', 'bao'])
print("[DEBUG] handle_tqo completed")
EOF

if [ $? -ne 0 ]; then
    echo "✗ ERROR: Training failed!"
    exit 1
fi

TRAIN_END=$(date +%s)
TRAIN_TIME=$((TRAIN_END - TRAIN_START))
echo "✓ Training completed in ${TRAIN_TIME}s"

# Ensure Bao server is stopped after training
echo "Ensuring Bao server is stopped..."
pkill -f "python.*bao_server.*main.py" || true
sleep 2

# Step 2: Test with Bao
echo ""
echo "[2/4] Testing with Bao optimizer..."
echo "----------------------------------------------------------------"
TEST_BAO_START=$(date +%s)

BAO_FORCE_CPU=1 python3 << EOF 2>&1 | tee $LOG_DIR/${TODAY}_test_bao_${DATASET}_${QUERY_SET}.log
import sys
import os
os.environ['BAO_FORCE_CPU'] = '1'  # Force CPU to avoid CUDA multiprocessing issues
sys.path.insert(0, '.')
from cli import GLOBAL_CONFIG, handle_iqo

GLOBAL_CONFIG['dataset'] = '$DATASET'
GLOBAL_CONFIG['query_set'] = '$QUERY_SET'
GLOBAL_CONFIG['pg_port'] = 5430
print(f"Dataset: {GLOBAL_CONFIG['dataset']}")
print(f"Query Set: {GLOBAL_CONFIG['query_set']}")
print(f"PostgreSQL Port: {GLOBAL_CONFIG['pg_port']}")
print("Executing: iqo bao bao")
handle_iqo(['iqo', 'bao', 'bao'])
EOF

if [ $? -ne 0 ]; then
    echo "✗ ERROR: Bao testing failed!"
    exit 1
fi

TEST_BAO_END=$(date +%s)
TEST_BAO_TIME=$((TEST_BAO_END - TEST_BAO_START))
echo "✓ Bao testing completed in ${TEST_BAO_TIME}s"

# Ensure Bao server is stopped after Bao testing
echo "Ensuring Bao server is stopped..."
pkill -f "python.*bao_server.*main.py" || true
sleep 2

# Step 3: Test with PostgreSQL (baseline)
echo ""
echo "[3/4] Testing with PostgreSQL optimizer..."
echo "----------------------------------------------------------------"
TEST_PG_START=$(date +%s)

python3 << EOF 2>&1 | tee $LOG_DIR/${TODAY}_test_pg_${DATASET}_${QUERY_SET}.log
import sys
sys.path.insert(0, '.')
from cli import GLOBAL_CONFIG, handle_iqo

GLOBAL_CONFIG['dataset'] = '$DATASET'
GLOBAL_CONFIG['query_set'] = '$QUERY_SET'
GLOBAL_CONFIG['pg_port'] = 5430
print(f"Dataset: {GLOBAL_CONFIG['dataset']}")
print(f"Query Set: {GLOBAL_CONFIG['query_set']}")
print(f"PostgreSQL Port: {GLOBAL_CONFIG['pg_port']}")
print("Executing: iqo bao pg")
handle_iqo(['iqo', 'bao', 'pg'])
EOF

if [ $? -ne 0 ]; then
    echo "✗ ERROR: PostgreSQL testing failed!"
    exit 1
fi

TEST_PG_END=$(date +%s)
TEST_PG_TIME=$((TEST_PG_END - TEST_PG_START))
echo "✓ PostgreSQL testing completed in ${TEST_PG_TIME}s"

# Final check: Ensure all Bao servers are stopped
echo "Final cleanup: Ensuring all Bao servers are stopped..."
pkill -f "python.*bao_server.*main.py" || true
sleep 2

# ============================================================
# Step 4: Save models and clean up for next run
# ============================================================
echo ""
echo "[4/4] Saving models and cleaning up for next run..."
echo "----------------------------------------------------------------"

# Get absolute path for SAVE_DIR before changing directory
SAVE_DIR="$(pwd)/$LOG_DIR/${TIMESTAMP}_${DATASET}_${QUERY_SET}"
mkdir -p "$SAVE_DIR"
echo "Saving artifacts to: $SAVE_DIR"

if [ -d "$BAO_SERVER_DIR" ]; then
    cd "$BAO_SERVER_DIR"

    # Move models to save directory (using absolute path)
    [ -e bao_default_model ] && mv bao_default_model "$SAVE_DIR/bao_default_model" && echo "✓ Saved bao_default_model" || echo "⚠ No bao_default_model found"
    [ -e bao_previous_model ] && mv bao_previous_model "$SAVE_DIR/bao_previous_model" && echo "✓ Saved bao_previous_model" || true
    [ -e bao_tmp_model ] && mv bao_tmp_model "$SAVE_DIR/bao_tmp_model" && echo "✓ Saved bao_tmp_model" || true

    # Save training and test logs
    [ -e bao_training_results.log ] && mv bao_training_results.log "$SAVE_DIR/bao_training_results.log" && echo "✓ Saved training log" || true
    [ -e bao_test_results.log ] && mv bao_test_results.log "$SAVE_DIR/bao_test_results.log" && echo "✓ Saved test log" || true

    # Remove experience database for clean next run
    [ -e bao.db ] && rm bao.db && echo "✓ Cleared experience database (bao.db)" || echo "⚠ No bao.db found"

    cd - > /dev/null

    echo ""
    echo "✓ All artifacts saved to: $SAVE_DIR"
    echo "✓ Environment cleaned - ready for next run"
else
    echo "⚠ Warning: Bao server directory not found at $BAO_SERVER_DIR"
fi

# Summary
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "================================================================"
echo "Pipeline Completed Successfully!"
echo "================================================================"
echo "Timing Summary:"
echo "  Training:           ${TRAIN_TIME}s ($(($TRAIN_TIME / 60))m)"
echo "  Bao Testing:        ${TEST_BAO_TIME}s ($(($TEST_BAO_TIME / 60))m)"
echo "  PostgreSQL Testing: ${TEST_PG_TIME}s ($(($TEST_PG_TIME / 60))m)"
echo "  Total:              ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Files saved in: $LOG_DIR/"
echo "  Models:      ${TIMESTAMP}_${DATASET}_${QUERY_SET}/"
echo "  Training:    ${TODAY}_train_${DATASET}_${QUERY_SET}.log"
echo "  Bao Test:    ${TODAY}_test_bao_${DATASET}_${QUERY_SET}.log"
echo "  PG Test:     ${TODAY}_test_pg_${DATASET}_${QUERY_SET}.log"
echo ""
echo "Next run will start with clean environment (bao.db cleared)"
echo ""
echo "End Time: $(date)"
echo "================================================================"
