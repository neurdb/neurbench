#!/bin/bash
#
# Bao Training and Testing Pipeline
# Usage: ./schedule_bao_simple.sh [QUERY_SET]
#        ./schedule_bao_simple.sh [OPTIONS]
#
# Simple mode (backward compatible):
#   ./schedule_bao_simple.sh                           # Use default query_set
#   ./schedule_bao_simple.sh job      # Use specified query_set
#
# Advanced mode (separate train/test configuration):
#   ./schedule_bao_simple.sh --train-dataset imdb_ori --train-query-set job-train \
#                            --test-dataset imdb_drift --test-query-set job-test
#
# Options:
#   --train-dataset DATASET      Dataset for training (default: imdb_ori)
#   --train-query-set QUERY_SET  Query set for training (default: job)
#   --test-dataset DATASET       Dataset for testing (default: same as train-dataset)
#   --test-query-set QUERY_SET   Query set for testing (default: same as train-query-set)
#   --force-retrain              Force retraining even if model exists
#   --force-retest               Force retesting even if test results exist
#   -h, --help                   Show this help message
#
# Note: The script automatically detects existing models and test results.
#       - If a model exists for train-dataset/train-query-set, it loads it.
#       - If test results exist for test-dataset/test-query-set, it skips testing.
#       Use --force-retrain or --force-retest to override.
#

set -e  # Exit on error

# Parse command line arguments - show_help function
show_help() {
    echo "Bao Training and Testing Pipeline"
    echo ""
    echo "Usage:"
    echo "  ./schedule_bao_simple.sh [QUERY_SET]"
    echo "  ./schedule_bao_simple.sh [OPTIONS]"
    echo ""
    echo "Simple mode (backward compatible):"
    echo "  ./schedule_bao_simple.sh                      # Use defaults"
    echo "  ./schedule_bao_simple.sh job # Use specified query_set"
    echo ""
    echo "Options:"
    echo "  --train-dataset DATASET      Dataset for training (default: imdb_ori)"
    echo "  --train-query-set QUERY_SET  Query set for training (default: job)"
    echo "  --test-dataset DATASET       Dataset for testing (default: same as train-dataset)"
    echo "  --test-query-set QUERY_SET   Query set for testing (default: same as train-query-set)"
    echo "  --force-retrain              Force retraining even if model exists"
    echo "  --force-retest               Force retesting even if test results exist"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Note: The script auto-detects existing models and test results."
    echo "      - Models: loaded from cache if train-dataset/query-set matches"
    echo "      - Tests: skipped if test-dataset/query-set results exist"
    echo "      Use --force-retrain or --force-retest to override."
    echo ""
    echo "Examples:"
    echo "  # Same dataset/query for train and test"
    echo "  ./schedule_bao_simple.sh job"
    echo ""
    echo "  # Different datasets for train and test"
    echo "  ./schedule_bao_simple.sh --train-dataset imdb_ori --test-dataset imdb_drift"
    echo ""
    echo "  # Fully customized"
    echo "  ./schedule_bao_simple.sh --train-dataset imdb_ori --train-query-set job-train \\"
    echo "                           --test-dataset imdb_drift --test-query-set job-test"
    exit 0
}

# Handle --help before anything else
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
fi

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

# Default configuration
TRAIN_DATASET="imdb_ori"
TRAIN_QUERY_SET="job"
TEST_DATASET=""
TEST_QUERY_SET=""
FORCE_RETRAIN=false
FORCE_RETEST=false

# Check if using advanced mode (any argument starts with --)
if [[ "$1" == -* ]]; then
    # Advanced mode: parse named arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --train-dataset)
                TRAIN_DATASET="$2"
                shift 2
                ;;
            --train-query-set)
                TRAIN_QUERY_SET="$2"
                shift 2
                ;;
            --test-dataset)
                TEST_DATASET="$2"
                shift 2
                ;;
            --test-query-set)
                TEST_QUERY_SET="$2"
                shift 2
                ;;
            --force-retrain)
                FORCE_RETRAIN=true
                shift
                ;;
            --force-retest)
                FORCE_RETEST=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
else
    # Simple mode: first argument is QUERY_SET (backward compatible)
    if [ -n "$1" ]; then
        TRAIN_QUERY_SET="$1"
    fi
fi

# Set defaults for test if not specified
if [ -z "$TEST_DATASET" ]; then
    TEST_DATASET="$TRAIN_DATASET"
fi
if [ -z "$TEST_QUERY_SET" ]; then
    TEST_QUERY_SET="$TRAIN_QUERY_SET"
fi

# Configuration
LOG_DIR="./bao_logs_all"
BAO_SERVER_DIR="benchmarks/lqos/bao/bao_server"
TODAY=$(date +%Y%m%d)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================"
echo "Bao Training and Testing Pipeline"
echo "Training:"
echo "  Dataset:   $TRAIN_DATASET"
echo "  Query Set: $TRAIN_QUERY_SET"
if [ "$FORCE_RETRAIN" = true ]; then
echo "  Mode:      Force retrain (--force-retrain)"
else
echo "  Mode:      Auto (load existing model if available)"
fi
echo "Testing:"
echo "  Dataset:   $TEST_DATASET"
echo "  Query Set: $TEST_QUERY_SET"
echo "Start Time: $(date)"
echo "================================================================"

# Create log directory
mkdir -p $LOG_DIR

# Timing
START_TIME=$(date +%s)

# Step 1: Train Bao (or load existing model)
echo ""
echo "[1/4] Training Bao with $TRAIN_DATASET..."
echo "----------------------------------------------------------------"

TRAIN_TIME_START=$(date +%s)
SKIP_TRAINING=false

# Check if a trained model already exists for this dataset and query_set
MODEL_PATTERN="${TRAIN_DATASET}_${TRAIN_QUERY_SET}"
EXISTING_MODEL_DIR=$(ls -dt ${LOG_DIR}/*_${MODEL_PATTERN} 2>/dev/null | head -1)

if [ "$FORCE_RETRAIN" = true ]; then
    echo "--force-retrain specified, will retrain model"
    EXISTING_MODEL_DIR=""
fi

if [ -n "$EXISTING_MODEL_DIR" ] && [ -d "$EXISTING_MODEL_DIR/bao_default_model" ]; then
    echo "✓ Found existing model: $EXISTING_MODEL_DIR"
    echo "  Loading model instead of training..."

    # Clean up any existing models in bao_server
    rm -rf "${BAO_SERVER_DIR}/bao_default_model" 2>/dev/null || true
    rm -rf "${BAO_SERVER_DIR}/bao_previous_model" 2>/dev/null || true
    rm -rf "${BAO_SERVER_DIR}/bao_tmp_model" 2>/dev/null || true

    # Copy the saved model to bao_server directory
    cp -r "$EXISTING_MODEL_DIR/bao_default_model" "${BAO_SERVER_DIR}/"
    if [ -d "$EXISTING_MODEL_DIR/bao_previous_model" ]; then
        cp -r "$EXISTING_MODEL_DIR/bao_previous_model" "${BAO_SERVER_DIR}/"
    fi

    echo "✓ Model loaded from: $EXISTING_MODEL_DIR"
    SKIP_TRAINING=true
    TRAIN_TIME=0
else
    echo "No existing model found for ${MODEL_PATTERN}"
    echo "Starting training..."
    echo "[DEBUG] Query directory: queries/$TRAIN_QUERY_SET/train"
    echo "[DEBUG] Database: $TRAIN_DATASET"
    echo "[DEBUG] Port: 5430"

python3 -u  << EOF 2>&1 | tee $LOG_DIR/${TODAY}_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log
import sys
import os
print("[DEBUG] Python script started")
sys.path.insert(0, '.')
print("[DEBUG] Importing cli module...")
from cli import GLOBAL_CONFIG, handle_tqo
print("[DEBUG] cli module imported")

GLOBAL_CONFIG['dataset'] = '$TRAIN_DATASET'
GLOBAL_CONFIG['query_set'] = '$TRAIN_QUERY_SET'
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

    TRAIN_TIME_END=$(date +%s)
    TRAIN_TIME=$((TRAIN_TIME_END - TRAIN_TIME_START))
    echo "✓ Training completed in ${TRAIN_TIME}s"
fi

# Ensure Bao server is stopped after training
echo "Ensuring Bao server is stopped..."
pkill -f "python3.*main.py" || true
sleep 2

# Step 2: Test with Bao
echo ""
echo "[2/4] Testing with Bao optimizer..."
echo "----------------------------------------------------------------"

TEST_BAO_START=$(date +%s)
SKIP_BAO_TEST=false

# Check if Bao test results already exist
EXISTING_BAO_TEST=$(ls -t ${LOG_DIR}/*_test_bao_${TEST_DATASET}_${TEST_QUERY_SET}.log 2>/dev/null | head -1)

if [ "$FORCE_RETEST" = true ]; then
    echo "--force-retest specified, will run Bao test"
    EXISTING_BAO_TEST=""
fi

if [ -n "$EXISTING_BAO_TEST" ] && [ -s "$EXISTING_BAO_TEST" ]; then
    echo "✓ Found existing Bao test results: $EXISTING_BAO_TEST"
    echo "  Skipping Bao test (use --force-retest to override)"
    SKIP_BAO_TEST=true
    TEST_BAO_TIME=0
else
    echo "[DEBUG] Query directory: queries/$TEST_QUERY_SET/test"
    echo "[DEBUG] Database: $TEST_DATASET"

python3 -u  -u  << EOF 2>&1 | tee $LOG_DIR/${TODAY}_test_bao_${TEST_DATASET}_${TEST_QUERY_SET}.log
import sys
import os
sys.path.insert(0, '.')
from cli import GLOBAL_CONFIG, handle_iqo

GLOBAL_CONFIG['dataset'] = '$TEST_DATASET'
GLOBAL_CONFIG['query_set'] = '$TEST_QUERY_SET'
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
fi

# Ensure Bao server is stopped after Bao testing
echo "Ensuring Bao server is stopped..."
pkill -f "python3.*main.py" || true
sleep 2

# Step 3: Test with PostgreSQL (baseline)
echo ""
echo "[3/4] Testing with PostgreSQL optimizer..."
echo "----------------------------------------------------------------"

TEST_PG_START=$(date +%s)
SKIP_PG_TEST=false

# Check if PostgreSQL test results already exist
EXISTING_PG_TEST=$(ls -t ${LOG_DIR}/*_test_pg_${TEST_DATASET}_${TEST_QUERY_SET}.log 2>/dev/null | head -1)

if [ "$FORCE_RETEST" = true ]; then
    echo "--force-retest specified, will run PostgreSQL test"
    EXISTING_PG_TEST=""
fi

if [ -n "$EXISTING_PG_TEST" ] && [ -s "$EXISTING_PG_TEST" ]; then
    echo "✓ Found existing PostgreSQL test results: $EXISTING_PG_TEST"
    echo "  Skipping PostgreSQL test (use --force-retest to override)"
    SKIP_PG_TEST=true
    TEST_PG_TIME=0
else
    echo "[DEBUG] Query directory: queries/$TEST_QUERY_SET/test"
    echo "[DEBUG] Database: $TEST_DATASET"

python3 -u  << EOF 2>&1 | tee $LOG_DIR/${TODAY}_test_pg_${TEST_DATASET}_${TEST_QUERY_SET}.log
import sys
sys.path.insert(0, '.')
from cli import GLOBAL_CONFIG, handle_iqo

GLOBAL_CONFIG['dataset'] = '$TEST_DATASET'
GLOBAL_CONFIG['query_set'] = '$TEST_QUERY_SET'
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
fi

# Final check: Ensure all Bao servers are stopped
echo "Final cleanup: Ensuring all Bao servers are stopped..."
pkill -f "python3.*main.py" || true
sleep 2

# ============================================================
# Step 4: Save models and clean up for next run
# ============================================================
echo ""
echo "[4/4] Saving models and cleaning up for next run..."
echo "----------------------------------------------------------------"

if [ "$SKIP_TRAINING" = true ]; then
    echo "Model was loaded from existing cache, skipping model save."
    echo "Model source: $EXISTING_MODEL_DIR"
    SAVE_DIR="$EXISTING_MODEL_DIR"
else
    # Get absolute path for SAVE_DIR before changing directory
    SAVE_DIR="$(pwd)/$LOG_DIR/${TIMESTAMP}_${TRAIN_DATASET}_${TRAIN_QUERY_SET}"
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
echo "Configuration:"
echo "  Training:   Dataset=$TRAIN_DATASET, QuerySet=$TRAIN_QUERY_SET"
echo "  Testing:    Dataset=$TEST_DATASET, QuerySet=$TEST_QUERY_SET"
if [ "$SKIP_TRAINING" = true ]; then
echo "  Model:      Loaded from cache"
echo "              $EXISTING_MODEL_DIR"
else
echo "  Model:      Newly trained"
fi
echo ""
echo "Timing Summary:"
if [ "$SKIP_TRAINING" = true ]; then
echo "  Model Load:         (cached)"
else
echo "  Training:           ${TRAIN_TIME}s ($(($TRAIN_TIME / 60))m)"
fi
if [ "$SKIP_BAO_TEST" = true ]; then
echo "  Bao Testing:        (cached)"
else
echo "  Bao Testing:        ${TEST_BAO_TIME}s ($(($TEST_BAO_TIME / 60))m)"
fi
if [ "$SKIP_PG_TEST" = true ]; then
echo "  PostgreSQL Testing: (cached)"
else
echo "  PostgreSQL Testing: ${TEST_PG_TIME}s ($(($TEST_PG_TIME / 60))m)"
fi
echo "  Total:              ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Files in: $LOG_DIR/"
echo "  Models:    $SAVE_DIR"
if [ "$SKIP_BAO_TEST" = true ]; then
echo "  Bao Test:  $EXISTING_BAO_TEST (cached)"
else
echo "  Bao Test:  ${TODAY}_test_bao_${TEST_DATASET}_${TEST_QUERY_SET}.log"
fi
if [ "$SKIP_PG_TEST" = true ]; then
echo "  PG Test:   $EXISTING_PG_TEST (cached)"
else
echo "  PG Test:   ${TODAY}_test_pg_${TEST_DATASET}_${TEST_QUERY_SET}.log"
fi
echo ""
echo "End Time: $(date)"
echo "================================================================"
