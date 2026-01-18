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
#   ./schedule_balsa_simple.sh --train-dataset imdb --train-query-set job \
#                              --test-dataset imdb_drift --test-query-set job
#
# Options:
#   --train-dataset DATASET      Dataset for training (default: imdb)
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
set -o pipefail  # Capture errors in pipelines (e.g., python3 | tee)

# Activate conda environment (for docker exec -c mode)
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate ai4db_new
    export PATH="/root/miniconda3/envs/ai4db_new/bin:$PATH"
fi

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
    echo "  --train-dataset DATASET      Dataset for training (default: imdb)"
    echo "  --train-query-set QUERY_SET  Query set for training (default: job)"
    echo "  --test-dataset DATASET       Dataset for testing (default: same as train-dataset)"
    echo "  --test-query-set QUERY_SET   Query set for testing (default: same as train-query-set)"
    echo "  --force-retrain              Force retraining even if model exists"
    echo "  --force-retest               Force retesting even if test results exist"
    echo "  --continue-training          Continue training from existing model and data"
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
    echo "  ./schedule_balsa_simple.sh --train-dataset imdb --test-dataset imdb_drift"
    echo ""
    echo "  # Fully customized"
    echo "  ./schedule_balsa_simple.sh --train-dataset imdb --train-query-set job \\"
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
TRAIN_DATASET="imdb"
TRAIN_QUERY_SET="job"
TEST_DATASET=""
TEST_QUERY_SET=""
FORCE_RETRAIN=false
FORCE_RETEST=false
CONTINUE_TRAINING=false

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
            --continue-training)
                CONTINUE_TRAINING=true
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

# Model checkpoint pattern for Balsa (with timestamp)
MODEL_PREFIX="${TIMESTAMP}_${TRAIN_DATASET}_${TRAIN_QUERY_SET}"

# Find existing checkpoint - Balsa saves to: {LOG_DIR}/balsa_{dataset}_{query_set}_full_checkpoint.pt
# After training, it's moved to: {LOG_DIR}/{timestamp}_{dataset}_{query_set}/checkpoint.pt
MODEL_PREFIX_PATTERN="balsa_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_full"
DIRECT_CHECKPOINT="${LOG_DIR}/${MODEL_PREFIX_PATTERN}_checkpoint.pt"
CHECKPOINT_DIR_PATTERN="${LOG_DIR}/${MODEL_PREFIX_PATTERN}_checkpoints"

# Find the latest model directory for data/replay buffers
LATEST_MODEL_DIR=$(ls -dt ${LOG_DIR}/train_${TRAIN_DATASET}_${TRAIN_QUERY_SET} ${LOG_DIR}/*_${TRAIN_DATASET}_${TRAIN_QUERY_SET} 2>/dev/null | grep -v "_test_" | head -1 || true)
if [ -n "$LATEST_MODEL_DIR" ]; then
    echo "Found model directory: $LATEST_MODEL_DIR"
fi

# Find checkpoint: first check direct location, then search in timestamped directory
if [ -f "$DIRECT_CHECKPOINT" ]; then
    CHECKPOINT_FILE="$DIRECT_CHECKPOINT"
    echo "Found checkpoint: $CHECKPOINT_FILE"
elif [ -n "$LATEST_MODEL_DIR" ]; then
    CHECKPOINT_FILE=$(find "$LATEST_MODEL_DIR" -name "checkpoint.pt" -type f 2>/dev/null | head -1 || true)
    if [ -n "$CHECKPOINT_FILE" ]; then
        echo "Found checkpoint: $CHECKPOINT_FILE"
    fi
else
    CHECKPOINT_FILE=""
fi

if [ "$FORCE_RETRAIN" = true ]; then
    echo "--force-retrain specified, will retrain model"
    EXISTING_MODEL=""
elif [ -n "$CHECKPOINT_FILE" ] && [ -f "$CHECKPOINT_FILE" ]; then
    EXISTING_MODEL="$CHECKPOINT_FILE"
else
    EXISTING_MODEL=""
fi

# Find existing replay buffers for continue training
# Check both direct logs dir and timestamped dir
EXISTING_REPLAY_BUFFERS=""
REPLAY_BUFFER_LOCATION=""
# First check direct logs dir (new real-time save location)
if [ -d "${LOG_DIR}/data" ]; then
    EXISTING_REPLAY_BUFFERS=$(ls "${LOG_DIR}/data"/replay-*.pkl 2>/dev/null | head -1 || true)
    if [ -n "$EXISTING_REPLAY_BUFFERS" ]; then
        REPLAY_BUFFER_LOCATION="${LOG_DIR}/data/"
    fi
fi
# If not found, check timestamped training dir
if [ -z "$EXISTING_REPLAY_BUFFERS" ] && [ -n "$LATEST_MODEL_DIR" ] && [ -d "$LATEST_MODEL_DIR/data" ]; then
    EXISTING_REPLAY_BUFFERS=$(ls "$LATEST_MODEL_DIR/data"/replay-*.pkl 2>/dev/null | head -1 || true)
    if [ -n "$EXISTING_REPLAY_BUFFERS" ]; then
        REPLAY_BUFFER_LOCATION="$LATEST_MODEL_DIR/data/"
    fi
fi

if [ -n "$EXISTING_MODEL" ] && [ "$CONTINUE_TRAINING" = false ]; then
    echo "Found existing model: $EXISTING_MODEL"
    echo "  Skipping training (use --force-retrain to override or --continue-training to continue)"
    SKIP_TRAINING=true
    TRAIN_TIME=0
elif [ -n "$EXISTING_MODEL" ] && [ "$CONTINUE_TRAINING" = true ]; then
    echo "Continue training mode enabled"
    echo "Found existing model: $EXISTING_MODEL"
    if [ -n "$EXISTING_REPLAY_BUFFERS" ]; then
        echo "Found existing replay buffers in: $REPLAY_BUFFER_LOCATION"
    fi
    # Fall through to training section below
fi

# Run training if not skipping
if [ "$SKIP_TRAINING" = false ]; then
    if [ "$CONTINUE_TRAINING" = true ] && [ -n "$EXISTING_MODEL" ]; then
        echo "Continuing training from existing model..."
    else
        echo "No existing model found for ${TRAIN_DATASET}_${TRAIN_QUERY_SET}"
        echo "Starting training (will save as ${MODEL_PREFIX})..."
    fi

    # Update config with training database
    update_db_config "$TRAIN_DATASET"

    # Change to Balsa directory
    cd "$BALSA_DIR"

    # Clean up old/failed training artifacts (but preserve for continue training)
    echo "Cleaning up old training artifacts..."
    if [ "$CONTINUE_TRAINING" = true ] && [ -n "$EXISTING_MODEL" ]; then
        # For continue training, copy existing replay buffers to current data dir
        rm -rf logs tensorboard_logs runs
        mkdir -p logs data

        # Copy replay buffer from the location found earlier
        if [ -n "$REPLAY_BUFFER_LOCATION" ]; then
            echo "Copying replay buffer from: ${PROJECT_ROOT}/${REPLAY_BUFFER_LOCATION}"
            cp "${PROJECT_ROOT}/${REPLAY_BUFFER_LOCATION}"replay-*.pkl data/ 2>/dev/null || true
        else
            echo "Warning: No replay buffer found for continue training"
        fi

        # Copy initial_policy_data from multiple possible locations
        cp "${PROJECT_ROOT}/${LOG_DIR}/data"/initial_policy_data*.pkl data/ 2>/dev/null || true
        cp "${PROJECT_ROOT}/${LATEST_MODEL_DIR}/data"/initial_policy_data*.pkl data/ 2>/dev/null || true
    else
        rm -rf logs tensorboard_logs data runs
        mkdir -p logs
    fi

    # Install packages
    echo "Installing Balsa packages..."
    pip install -e . > /dev/null 2>&1
    pip install -e pg_executor > /dev/null 2>&1

    # Prewarm database once before training
    echo "Prewarming database before training..."
    python3 -c "
import sys
sys.path.insert(0, 'pg_executor')
from pg_executor.pg_executor import prewarm_database
prewarm_database()
"

    # Build continue training arguments
    CONTINUE_ARGS=""
    if [ "$CONTINUE_TRAINING" = true ] && [ -n "$EXISTING_MODEL" ]; then
        CONTINUE_ARGS="--agent_checkpoint ${PROJECT_ROOT}/${EXISTING_MODEL}"
        if [ -d "data" ] && ls data/replay-*.pkl 1>/dev/null 2>&1; then
            CONTINUE_ARGS="$CONTINUE_ARGS --prev_replay_buffers data/replay-*.pkl"
        fi

        # Find the last completed iteration from checkpoint filenames
        # Checkpoints are in: {LOG_DIR}/balsa_{dataset}_{query_set}_full_checkpoints/{timestamp}/checkpoint__iter{N}.pt
        LAST_ITER=0
        ITER_CHECKPOINT_DIR="${PROJECT_ROOT}/${LOG_DIR}/${MODEL_PREFIX_PATTERN}_checkpoints"
        if [ -d "$ITER_CHECKPOINT_DIR" ]; then
            echo "Looking for iterations in: $ITER_CHECKPOINT_DIR"
            for ckpt in "$ITER_CHECKPOINT_DIR"/*/checkpoint__iter*.pt; do
                if [ -f "$ckpt" ]; then
                    # Extract iteration number from filename
                    iter_num=$(basename "$ckpt" | sed 's/checkpoint__iter\([0-9]*\)\.pt/\1/')
                    if [ "$iter_num" -gt "$LAST_ITER" ] 2>/dev/null; then
                        LAST_ITER=$iter_num
                    fi
                fi
            done
        fi

        if [ "$LAST_ITER" -gt 0 ]; then
            START_ITER=$((LAST_ITER + 1))
            CONTINUE_ARGS="$CONTINUE_ARGS --start_iter $START_ITER"
            echo "Found last completed iteration: $LAST_ITER, resuming from iteration $START_ITER"
        fi

        echo "Continue training args: $CONTINUE_ARGS"
    fi

    # Run training
    export WANDB_MODE=disabled
    export PYTHONUNBUFFERED=1
    export BALSA_EXECUTION_MODE=single  # Training mode (1 run, prewarm already done)
    export BALSA_SKIP_PREWARM=1  # Skip per-query prewarm since we did it once above
    export BALSA_TIMESTAMP=$TIMESTAMP  # Pass timestamp to Python for consistent directory naming

    if [ "$CONTINUE_TRAINING" = true ] && [ -n "$EXISTING_MODEL" ]; then
        # Continue training: only run once (baseline already collected, just continue RL training)
        echo "Running Balsa continue training..."
        CUDA_VISIBLE_DEVICES=3 python3 -u run.py \
            --run NB_Balsa_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_datashift \
            --local \
            $CONTINUE_ARGS \
            2>&1 | tee "${PROJECT_ROOT}/${LOG_DIR}/${TIMESTAMP}_train_continue_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log"
        TRAIN_RESULT=$?
    else
        # Fresh training: run twice
        # 1st run: collects baseline (expert) data if initial_policy_data.pkl doesn't exist
        # 2nd run: actual RL training using the collected baseline
        echo "Running Balsa training (iteration 1/2 - baseline collection)..."
        CUDA_VISIBLE_DEVICES=3 python3 -u run.py \
            --run NB_Balsa_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_datashift \
            --local \
            $CONTINUE_ARGS \
            2>&1 | tee "${PROJECT_ROOT}/${LOG_DIR}/${TIMESTAMP}_train_1_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log"

        echo "Running Balsa training (iteration 2/2 - RL training)..."
        CUDA_VISIBLE_DEVICES=3 python3 -u run.py \
            --run NB_Balsa_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_datashift \
            --local \
            $CONTINUE_ARGS \
            2>&1 | tee "${PROJECT_ROOT}/${LOG_DIR}/${TIMESTAMP}_train_2_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log"
        TRAIN_RESULT=$?
    fi

    # Save training artifacts with timestamp
    sleep 10
    TRAIN_SAVE_DIR="${PROJECT_ROOT}/${LOG_DIR}/${MODEL_PREFIX}"
    mkdir -p "$TRAIN_SAVE_DIR"
    echo "Saving training artifacts to: $TRAIN_SAVE_DIR"

    [ -d tensorboard_logs ] && mv tensorboard_logs "$TRAIN_SAVE_DIR/" && echo "Saved tensorboard_logs"
    [ -d runs ] && mv runs "$TRAIN_SAVE_DIR/" && echo "Saved runs"
    [ -d logs ] && mv logs "$TRAIN_SAVE_DIR/" && echo "Saved logs"

    # Move data from balsa_logs_all/data/ to timestamped directory (real-time saved data)
    # Python saves to both balsa_logs_all/data/ and local data/, we move one and clean the other
    BALSA_DATA_DIR="${PROJECT_ROOT}/${LOG_DIR}/data"
    if [ -d "$BALSA_DATA_DIR" ]; then
        mv "$BALSA_DATA_DIR" "$TRAIN_SAVE_DIR/" && echo "Moved data from $BALSA_DATA_DIR"
        # Clean up local backup
        [ -d data ] && rm -rf data
    elif [ -d data ]; then
        # Fallback: use local data/ if balsa_logs_all/data/ doesn't exist
        mv data "$TRAIN_SAVE_DIR/" && echo "Saved local data"
    fi

    # Move checkpoint to timestamped directory
    # Balsa saves checkpoint to: balsa_logs_all/balsa_{dataset}_{query_set}_full_checkpoint.pt
    BALSA_CHECKPOINT="${PROJECT_ROOT}/${LOG_DIR}/balsa_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_full_checkpoint.pt"
    if [ -f "$BALSA_CHECKPOINT" ]; then
        mv "$BALSA_CHECKPOINT" "$TRAIN_SAVE_DIR/checkpoint.pt"
        echo "Moved checkpoint to $TRAIN_SAVE_DIR/checkpoint.pt"
        CHECKPOINT_FILE="$TRAIN_SAVE_DIR/checkpoint.pt"
    else
        echo "Warning: No checkpoint file found after training"
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

# Check if Balsa test results already exist (find latest)
TEST_RESULT_DIR="${LOG_DIR}/${TIMESTAMP}_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_test_${TEST_DATASET}_${TEST_QUERY_SET}"
LATEST_TEST_DIR=$(ls -dt ${LOG_DIR}/*_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_test_${TEST_DATASET}_${TEST_QUERY_SET} 2>/dev/null | head -1 || true)
EXISTING_BALSA_TEST=""

if [ "$FORCE_RETEST" = true ]; then
    echo "--force-retest specified, will run Balsa test"
elif [ -n "$LATEST_TEST_DIR" ] && [ -d "$LATEST_TEST_DIR" ] && [ "$(ls -A $LATEST_TEST_DIR 2>/dev/null)" ]; then
    EXISTING_BALSA_TEST="$LATEST_TEST_DIR"
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

    # Clean up old artifacts
    rm -rf logs data tensorboard_logs runs
    mkdir -p logs

    # Find the latest training directory (for testing, we need the most recent model, exclude test directories)
    LATEST_TRAIN_DIR=$(ls -dt ${PROJECT_ROOT}/${LOG_DIR}/*_${TRAIN_DATASET}_${TRAIN_QUERY_SET} 2>/dev/null | grep -v "_train_" | head -1 || true)
    if [ -z "$LATEST_TRAIN_DIR" ]; then
        echo "ERROR: No training directory found for ${TRAIN_DATASET}_${TRAIN_QUERY_SET}"
        exit 1
    fi
    echo "Using training directory: $LATEST_TRAIN_DIR"

    # Copy data directory for testing (needed for workload info)
    if [ -d "$LATEST_TRAIN_DIR/data" ]; then
        echo "Copying data from $LATEST_TRAIN_DIR..."
        cp -r "$LATEST_TRAIN_DIR/data" ./
    fi

    # Install packages
    echo "Installing Balsa packages..."
    pip install -e . > /dev/null 2>&1
    pip install -e pg_executor > /dev/null 2>&1

    # Find checkpoint file from the latest training directory
    CHECKPOINT_FILE=$(find "$LATEST_TRAIN_DIR" -name "checkpoint.pt" -type f 2>/dev/null | head -1 || true)
    if [ -z "$CHECKPOINT_FILE" ] || [ ! -f "$CHECKPOINT_FILE" ]; then
        echo "ERROR: No trained model found in $LATEST_TRAIN_DIR"
        exit 1
    fi
    echo "Using checkpoint: $CHECKPOINT_FILE"

    # Prewarm database once before testing
    echo "Prewarming database before testing..."
    python3 -c "
import sys
sys.path.insert(0, 'pg_executor')
from pg_executor.pg_executor import prewarm_database
prewarm_database()
"

    # Run test
    echo "Running Balsa test..."
    export WANDB_MODE=disabled
    export PYTHONUNBUFFERED=1
    export BALSA_EXECUTION_MODE=single  # Testing mode (1 run, prewarm already done)
    export BALSA_SKIP_PREWARM=1  # Skip per-query prewarm since we did it once above
    CUDA_VISIBLE_DEVICES=3 python3 -u test_model.py \
        --run NB_Balsa_test_${TEST_DATASET}_${TEST_QUERY_SET}_datashift \
        --model_checkpoint "$CHECKPOINT_FILE" \
        2>&1 | tee "${PROJECT_ROOT}/${LOG_DIR}/${TIMESTAMP}_test_balsa_${TEST_DATASET}_${TEST_QUERY_SET}.log"

    TEST_RESULT=$?

    # Save test logs only (not training artifacts)
    sleep 5
    mkdir -p "${PROJECT_ROOT}/${TEST_RESULT_DIR}"

    [ -d logs ] && mv logs "${PROJECT_ROOT}/${TEST_RESULT_DIR}/" && echo "Saved test logs"

    # Clean up copied training data
    rm -rf data tensorboard_logs runs

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
SECS=$((TOTAL_TIME % 60))

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
echo "  Total:              ${HOURS}h ${MINUTES}m ${SECS}s"
echo ""
echo "Files in: $LOG_DIR/"
echo "  Model:      ${CHECKPOINT_FILE}"
if [ "$SKIP_BALSA_TEST" = true ]; then
echo "  Balsa Test: $EXISTING_BALSA_TEST (cached)"
else
echo "  Balsa Test: ${TIMESTAMP}_test_balsa_${TEST_DATASET}_${TEST_QUERY_SET}.log"
fi
echo ""
echo "End Time: $(date)"
echo "================================================================"
