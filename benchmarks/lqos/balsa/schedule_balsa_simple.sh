#!/bin/bash
#
# Balsa Training and Testing Pipeline
# Usage: ./schedule_balsa_simple.sh [QUERY_SET]
#        ./schedule_balsa_simple.sh [OPTIONS]
#
# Simple mode (backward compatible):
#   ./schedule_balsa_simple.sh                           # Use default query_set
#   ./schedule_balsa_simple.sh job                       # Use specified query_set
#
# Advanced mode (separate train/test configuration):
#   ./schedule_balsa_simple.sh --train-dataset imdb_ori --train-query-set job \
#                              --test-dataset imdb_drift --test-query-set job
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
    echo "Balsa Training and Testing Pipeline"
    echo ""
    echo "Usage:"
    echo "  ./schedule_balsa_simple.sh [QUERY_SET]"
    echo "  ./schedule_balsa_simple.sh [OPTIONS]"
    echo ""
    echo "Simple mode (backward compatible):"
    echo "  ./schedule_balsa_simple.sh                      # Use defaults"
    echo "  ./schedule_balsa_simple.sh job                  # Use specified query_set"
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
    echo "  ./schedule_balsa_simple.sh job"
    echo ""
    echo "  # Different datasets for train and test"
    echo "  ./schedule_balsa_simple.sh --train-dataset imdb_ori --test-dataset imdb_drift"
    echo ""
    echo "  # Fully customized"
    echo "  ./schedule_balsa_simple.sh --train-dataset imdb_ori --train-query-set job \\"
    echo "                             --test-dataset imdb_drift --test-query-set job"
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

# Balsa directories
BALSA_DIR="benchmarks/lqos/balsa"
PG_EXECUTOR_CONFIG="$BALSA_DIR/pg_executor/pg_executor/pg_executor.py"

# Verify Balsa exists
if [ ! -d "$BALSA_DIR" ]; then
    echo "ERROR: Balsa directory not found at $BALSA_DIR"
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
LOG_DIR="./balsa_logs_all"
TODAY=$(date +%Y%m%d)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PG_PORT=5433

echo "================================================================"
echo "Balsa Training and Testing Pipeline"
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

# Function to update pg_executor.py with database name
update_db_config() {
    local db_name="$1"
    echo "Updating pg_executor.py with DB=$db_name..."
    sed -i "s|LOCAL_DSN = \"postgres://postgres:postgres@172.17.0.1:${PG_PORT}/[^\"]*\"|LOCAL_DSN = \"postgres://postgres:postgres@172.17.0.1:${PG_PORT}/${db_name}\"|" "$PG_EXECUTOR_CONFIG"
}

# Function to initialize Conda environment
init_conda() {
    echo "Initializing Conda environment..."
    CONDA_BASE=${CONDA_BASE:-/root/miniconda3}
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "Warning: Conda initialization script not found at $CONDA_BASE/etc/profile.d/conda.sh"
        echo "Trying alternative locations..."
        for alt_path in /opt/miniconda3 /opt/conda ~/miniconda3 ~/anaconda3; do
            if [ -f "$alt_path/etc/profile.d/conda.sh" ]; then
                source "$alt_path/etc/profile.d/conda.sh"
                echo "Found Conda at $alt_path"
                break
            fi
        done
    fi

    # Try to activate the Balsa environment
    conda activate balsa_pre_17 2>/dev/null || {
        echo "Warning: Could not activate balsa_pre_17 environment"
        echo "Continuing with current Python environment..."
    }

    echo "Active Python: $(which python3)"
}

# Timing
START_TIME=$(date +%s)

# Initialize Conda
init_conda

# ============================================================
# Step 1: Train Balsa (or load existing model)
# ============================================================
echo ""
echo "[1/2] Training Balsa with $TRAIN_DATASET..."
echo "----------------------------------------------------------------"

TRAIN_TIME_START=$(date +%s)
SKIP_TRAINING=false

# Model checkpoint pattern for Balsa
MODEL_PREFIX="train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}"
CHECKPOINT_FILE="${LOG_DIR}/balsa_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_full_checkpoint.pt"

if [ "$FORCE_RETRAIN" = true ]; then
    echo "--force-retrain specified, will retrain model"
    EXISTING_MODEL=""
elif [ -f "$CHECKPOINT_FILE" ]; then
    EXISTING_MODEL="$CHECKPOINT_FILE"
else
    EXISTING_MODEL=""
fi

if [ -n "$EXISTING_MODEL" ]; then
    echo "Found existing model: $EXISTING_MODEL"
    echo "  Skipping training (use --force-retrain to override)"
    SKIP_TRAINING=true
    TRAIN_TIME=0
else
    echo "No existing model found for ${MODEL_PREFIX}"
    echo "Starting training..."

    # Update config with training database
    update_db_config "$TRAIN_DATASET"

    # Change to Balsa directory
    cd "$BALSA_DIR"

    # Clean up old logs
    rm -rf logs
    mkdir -p logs

    # Install packages
    echo "Installing Balsa packages..."
    pip install -e . > /dev/null 2>&1
    pip install -e pg_executor > /dev/null 2>&1

    # Run training (twice as per template)
    echo "Running Balsa training (iteration 1/2)..."
    export WANDB_MODE=disabled
    export PYTHONUNBUFFERED=1
    CUDA_VISIBLE_DEVICES=3 python3 -u run.py \
        --run NB_Balsa_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_datashift \
        --local \
        2>&1 | tee "${PROJECT_ROOT}/${LOG_DIR}/${TODAY}_train_1_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log"

    echo "Running Balsa training (iteration 2/2)..."
    CUDA_VISIBLE_DEVICES=3 python3 -u run.py \
        --run NB_Balsa_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_datashift \
        --local \
        2>&1 | tee "${PROJECT_ROOT}/${LOG_DIR}/${TODAY}_train_2_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log"

    TRAIN_RESULT=$?

    # Save training artifacts
    sleep 10
    TRAIN_SAVE_DIR="${PROJECT_ROOT}/${LOG_DIR}/${MODEL_PREFIX}"
    mkdir -p "$TRAIN_SAVE_DIR"

    [ -d tensorboard_logs ] && mv tensorboard_logs "$TRAIN_SAVE_DIR/" && echo "Saved tensorboard_logs"
    [ -d data ] && mv data "$TRAIN_SAVE_DIR/" && echo "Saved data"
    [ -d runs ] && mv runs "$TRAIN_SAVE_DIR/" && echo "Saved runs"
    [ -d logs ] && mv logs "$TRAIN_SAVE_DIR/" && echo "Saved logs"

    # Find and copy the checkpoint file
    LATEST_CHECKPOINT=$(find "$TRAIN_SAVE_DIR" -name "*.pt" -type f 2>/dev/null | head -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        cp "$LATEST_CHECKPOINT" "${PROJECT_ROOT}/${CHECKPOINT_FILE}"
        echo "Saved checkpoint to $CHECKPOINT_FILE"
    fi

    cd "$PROJECT_ROOT"

    if [ $TRAIN_RESULT -ne 0 ]; then
        echo "ERROR: Training failed!"
        exit 1
    fi

    TRAIN_TIME_END=$(date +%s)
    TRAIN_TIME=$((TRAIN_TIME_END - TRAIN_TIME_START))
    echo "Training completed in ${TRAIN_TIME}s"
fi

# ============================================================
# Step 2: Test with Balsa
# ============================================================
echo ""
echo "[2/2] Testing with Balsa optimizer..."
echo "----------------------------------------------------------------"

TEST_BALSA_START=$(date +%s)
SKIP_BALSA_TEST=false

# Check if Balsa test results already exist
TEST_RESULT_DIR="${LOG_DIR}/train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_test_${TEST_DATASET}_${TEST_QUERY_SET}"
EXISTING_BALSA_TEST=""

if [ "$FORCE_RETEST" = true ]; then
    echo "--force-retest specified, will run Balsa test"
elif [ -d "$TEST_RESULT_DIR" ] && [ "$(ls -A $TEST_RESULT_DIR 2>/dev/null)" ]; then
    EXISTING_BALSA_TEST="$TEST_RESULT_DIR"
fi

if [ -n "$EXISTING_BALSA_TEST" ]; then
    echo "Found existing Balsa test results: $EXISTING_BALSA_TEST"
    echo "  Skipping Balsa test (use --force-retest to override)"
    SKIP_BALSA_TEST=true
    TEST_BALSA_TIME=0
else
    # Update config with test database
    update_db_config "$TEST_DATASET"

    # Change to Balsa directory
    cd "$BALSA_DIR"

    # Clean up old logs
    rm -rf logs
    mkdir -p logs

    # Copy training artifacts back for testing
    TRAIN_SAVE_DIR="${PROJECT_ROOT}/${LOG_DIR}/${MODEL_PREFIX}"
    if [ -d "$TRAIN_SAVE_DIR" ]; then
        echo "Restoring training artifacts from $TRAIN_SAVE_DIR..."
        [ -d "$TRAIN_SAVE_DIR/tensorboard_logs" ] && cp -r "$TRAIN_SAVE_DIR/tensorboard_logs" ./
        [ -d "$TRAIN_SAVE_DIR/data" ] && cp -r "$TRAIN_SAVE_DIR/data" ./
        [ -d "$TRAIN_SAVE_DIR/runs" ] && cp -r "$TRAIN_SAVE_DIR/runs" ./
    fi

    # Install packages
    echo "Installing Balsa packages..."
    pip install -e . > /dev/null 2>&1
    pip install -e pg_executor > /dev/null 2>&1

    # Find checkpoint file
    if [ ! -f "${PROJECT_ROOT}/${CHECKPOINT_FILE}" ]; then
        echo "ERROR: No trained model found at ${PROJECT_ROOT}/${CHECKPOINT_FILE}"
        exit 1
    fi

    echo "Using model: ${PROJECT_ROOT}/${CHECKPOINT_FILE}"

    # Run test
    echo "Running Balsa test..."
    export WANDB_MODE=disabled
    export PYTHONUNBUFFERED=1
    CUDA_VISIBLE_DEVICES=3 python3 -u test_model.py \
        --run NB_Balsa_test_${TEST_DATASET}_${TEST_QUERY_SET}_datashift \
        --model_checkpoint "${PROJECT_ROOT}/${CHECKPOINT_FILE}" \
        2>&1 | tee "${PROJECT_ROOT}/${LOG_DIR}/${TODAY}_test_balsa_${TEST_DATASET}_${TEST_QUERY_SET}.log"

    TEST_RESULT=$?

    # Save test artifacts
    sleep 10
    mkdir -p "${PROJECT_ROOT}/${TEST_RESULT_DIR}"

    [ -d tensorboard_logs ] && mv tensorboard_logs "${PROJECT_ROOT}/${TEST_RESULT_DIR}/" && echo "Saved tensorboard_logs"
    [ -d data ] && mv data "${PROJECT_ROOT}/${TEST_RESULT_DIR}/" && echo "Saved data"
    [ -d runs ] && mv runs "${PROJECT_ROOT}/${TEST_RESULT_DIR}/" && echo "Saved runs"
    [ -d logs ] && mv logs "${PROJECT_ROOT}/${TEST_RESULT_DIR}/" && echo "Saved logs"

    cd "$PROJECT_ROOT"

    if [ $TEST_RESULT -ne 0 ]; then
        echo "ERROR: Balsa testing failed!"
        exit 1
    fi

    TEST_BALSA_END=$(date +%s)
    TEST_BALSA_TIME=$((TEST_BALSA_END - TEST_BALSA_START))
    echo "Balsa testing completed in ${TEST_BALSA_TIME}s"
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
echo "              $CHECKPOINT_FILE"
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
if [ "$SKIP_BALSA_TEST" = true ]; then
echo "  Balsa Testing:      (cached)"
else
echo "  Balsa Testing:      ${TEST_BALSA_TIME}s ($(($TEST_BALSA_TIME / 60))m)"
fi
echo "  Total:              ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Files in: $LOG_DIR/"
echo "  Model:      ${CHECKPOINT_FILE}"
if [ "$SKIP_BALSA_TEST" = true ]; then
echo "  Balsa Test: $EXISTING_BALSA_TEST (cached)"
else
echo "  Balsa Test: ${TODAY}_test_balsa_${TEST_DATASET}_${TEST_QUERY_SET}.log"
fi
echo ""
echo "End Time: $(date)"
echo "================================================================"
