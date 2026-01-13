#!/bin/bash
#
# HybridQO Training and Testing Pipeline
# Usage: ./schedule_hybridqo_simple.sh [QUERY_SET]
#        ./schedule_hybridqo_simple.sh [OPTIONS]
#
# Simple mode (backward compatible):
#   ./schedule_hybridqo_simple.sh                           # Use default query_set
#   ./schedule_hybridqo_simple.sh job_full__train           # Use specified query_set
#
# Advanced mode (separate train/test configuration):
#   ./schedule_hybridqo_simple.sh --train-dataset imdb --train-query-set job_full__train \
#                                 --test-dataset imdb_drift --test-query-set job_full__train
#
# Options:
#   --train-dataset DATASET      Dataset for training (default: imdb)
#   --train-query-set QUERY_SET  Query set for training (default: job_full__train)
#   --test-dataset DATASET       Dataset for testing (default: same as train-dataset)
#   --test-query-set QUERY_SET   Query set for testing (default: same as train-query-set)
#   --epoch NUM                  Number of training epochs (default: 50)
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

# Activate conda environment if available (for docker exec compatibility)
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate ai4db_new
    export PATH="/root/miniconda3/envs/ai4db_new/bin:$PATH"
fi

# Parse command line arguments - show_help function
show_help() {
    echo "HybridQO Training and Testing Pipeline"
    echo ""
    echo "Usage:"
    echo "  ./schedule_hybridqo_simple.sh [QUERY_SET]"
    echo "  ./schedule_hybridqo_simple.sh [OPTIONS]"
    echo ""
    echo "Simple mode (backward compatible):"
    echo "  ./schedule_hybridqo_simple.sh                      # Use defaults"
    echo "  ./schedule_hybridqo_simple.sh job_full__train      # Use specified query_set"
    echo ""
    echo "Options:"
    echo "  --train-dataset DATASET      Dataset for training (default: imdb)"
    echo "  --train-query-set QUERY_SET  Query set for training (default: job_full__train)"
    echo "  --test-dataset DATASET       Dataset for testing (default: same as train-dataset)"
    echo "  --test-query-set QUERY_SET   Query set for testing (default: same as train-query-set)"
    echo "  --epoch NUM                  Number of training epochs (default: 50)"
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
    echo "  ./schedule_hybridqo_simple.sh job_full__train"
    echo ""
    echo "  # Different datasets for train and test"
    echo "  ./schedule_hybridqo_simple.sh --train-dataset imdb --test-dataset imdb_2017"
    echo ""
    echo "  # Fully customized"
    echo "  ./schedule_hybridqo_simple.sh --train-dataset imdb --train-query-set job_full__train \\"
    echo "                                --test-dataset imdb_2017 --test-query-set job_full__train"
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

# HybridQO directories
HYBRIDQO_DIR="benchmarks/lqos/hybrid_qo"
WORKLOAD_DIR="benchmarks/lqos/hybrid_qo/workload"

# Verify HybridQO exists
if [ ! -d "$HYBRIDQO_DIR" ]; then
    echo "ERROR: HybridQO directory not found at $HYBRIDQO_DIR"
    exit 1
fi

# Default configuration
TRAIN_DATASET="imdb"
TRAIN_QUERY_SET="job_full__train"
TEST_DATASET=""
TEST_QUERY_SET=""
EPOCH=50
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
            --epoch)
                EPOCH="$2"
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
LOG_DIR="./hybridqo_logs_all"
TODAY=$(date +%Y%m%d)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================"
echo "HybridQO Training and Testing Pipeline"
echo "Training:"
echo "  Dataset:   $TRAIN_DATASET"
echo "  Query Set: $TRAIN_QUERY_SET"
echo "  Epochs:    $EPOCH"
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

    # Try to activate the HybridQO environment
    conda activate ai4db_new 2>/dev/null || {
        echo "Warning: Could not activate ai4db_new environment"
        echo "Continuing with current Python environment..."
    }

    echo "Active Conda environment: $CONDA_DEFAULT_ENV"
    echo "Active Python: $(which python3)"
}

# Timing
START_TIME=$(date +%s)

# Initialize Conda
init_conda

# ============================================================
# Step 1: Train HybridQO (or load existing model)
# ============================================================
echo ""
echo "[1/2] Training HybridQO with $TRAIN_DATASET..."
echo "----------------------------------------------------------------"

TRAIN_TIME_START=$(date +%s)
SKIP_TRAINING=false

# Model directory pattern for HybridQO
MODEL_PREFIX="model_train_w_${TRAIN_DATASET}_query_${TRAIN_QUERY_SET}"
MODEL_SAVE_DIR="${LOG_DIR}/${MODEL_PREFIX}"

if [ "$FORCE_RETRAIN" = true ]; then
    echo "--force-retrain specified, will retrain model"
    EXISTING_MODEL=""
elif [ -d "$MODEL_SAVE_DIR" ] && [ "$(ls -A $MODEL_SAVE_DIR 2>/dev/null)" ]; then
    EXISTING_MODEL="$MODEL_SAVE_DIR"
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

    # Change to HybridQO directory
    cd "$HYBRIDQO_DIR"

    # Clean up old logs and model
    rm -rf logs model
    mkdir -p logs model

    # Check query file exists
    QUERY_FILE="${PROJECT_ROOT}/${WORKLOAD_DIR}/${TRAIN_QUERY_SET}.json"
    if [ ! -f "$QUERY_FILE" ]; then
        # Try alternate location
        QUERY_FILE="workload/${TRAIN_QUERY_SET}.json"
        if [ ! -f "$QUERY_FILE" ]; then
            echo "ERROR: Query file not found: ${TRAIN_QUERY_SET}.json"
            echo "  Searched in: ${PROJECT_ROOT}/${WORKLOAD_DIR}/ and workload/"
            exit 1
        fi
    fi

    echo "Using query file: $QUERY_FILE"

    # Run training
    echo "Running HybridQO training with $EPOCH epochs..."
    python3 run_mcts.py \
        --query_file "$QUERY_FILE" \
        --train_database "$TRAIN_DATASET" \
        --test_database "$TRAIN_DATASET" \
        --train_test train \
        --epoch $EPOCH \
        2>&1 | tee "${PROJECT_ROOT}/${LOG_DIR}/${TODAY}_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log"

    TRAIN_RESULT=$?

    # Save training artifacts
    sleep 10
    mkdir -p "${PROJECT_ROOT}/${MODEL_SAVE_DIR}"
    LOGS_SAVE_DIR="${PROJECT_ROOT}/${LOG_DIR}/logs_train_w_${TRAIN_DATASET}_query_${TRAIN_QUERY_SET}"
    mkdir -p "$LOGS_SAVE_DIR"

    [ -d model ] && mv model/* "${PROJECT_ROOT}/${MODEL_SAVE_DIR}/" 2>/dev/null && echo "Saved model"
    [ -d logs ] && mv logs/* "$LOGS_SAVE_DIR/" 2>/dev/null && echo "Saved logs"

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
# Step 2: Test with HybridQO
# ============================================================
echo ""
echo "[2/2] Testing with HybridQO optimizer..."
echo "----------------------------------------------------------------"

TEST_HYBRIDQO_START=$(date +%s)
SKIP_HYBRIDQO_TEST=false

# Check if HybridQO test results already exist
TEST_RESULT_DIR="${LOG_DIR}/logs_train_w_${TRAIN_DATASET}_query_${TRAIN_QUERY_SET}_test_w_${TEST_DATASET}_query_${TEST_QUERY_SET}"
EXISTING_HYBRIDQO_TEST=""

if [ "$FORCE_RETEST" = true ]; then
    echo "--force-retest specified, will run HybridQO test"
elif [ -d "$TEST_RESULT_DIR" ] && [ "$(ls -A $TEST_RESULT_DIR 2>/dev/null)" ]; then
    EXISTING_HYBRIDQO_TEST="$TEST_RESULT_DIR"
fi

if [ -n "$EXISTING_HYBRIDQO_TEST" ]; then
    echo "Found existing HybridQO test results: $EXISTING_HYBRIDQO_TEST"
    echo "  Skipping HybridQO test (use --force-retest to override)"
    SKIP_HYBRIDQO_TEST=true
    TEST_HYBRIDQO_TIME=0
else
    # Check if model exists
    if [ ! -d "${PROJECT_ROOT}/${MODEL_SAVE_DIR}" ] || [ -z "$(ls -A ${PROJECT_ROOT}/${MODEL_SAVE_DIR} 2>/dev/null)" ]; then
        echo "ERROR: No trained model found at ${PROJECT_ROOT}/${MODEL_SAVE_DIR}"
        exit 1
    fi

    # Change to HybridQO directory
    cd "$HYBRIDQO_DIR"

    # Clean up old logs and model, then restore model
    rm -rf model logs
    mkdir -p logs

    echo "Restoring model from ${PROJECT_ROOT}/${MODEL_SAVE_DIR}..."
    cp -r "${PROJECT_ROOT}/${MODEL_SAVE_DIR}" ./model

    # Check query file exists
    QUERY_FILE="${PROJECT_ROOT}/${WORKLOAD_DIR}/${TEST_QUERY_SET}.json"
    if [ ! -f "$QUERY_FILE" ]; then
        # Try alternate location
        QUERY_FILE="workload/${TEST_QUERY_SET}.json"
        if [ ! -f "$QUERY_FILE" ]; then
            echo "ERROR: Query file not found: ${TEST_QUERY_SET}.json"
            exit 1
        fi
    fi

    echo "Using query file: $QUERY_FILE"
    echo "Using model: ${PROJECT_ROOT}/${MODEL_SAVE_DIR}"

    # Run test
    echo "Running HybridQO test..."
    python3 run_mcts.py \
        --query_file "$QUERY_FILE" \
        --train_database "$TRAIN_DATASET" \
        --test_database "$TEST_DATASET" \
        --train_test test \
        2>&1 | tee "${PROJECT_ROOT}/${LOG_DIR}/${TODAY}_test_hybridqo_${TEST_DATASET}_${TEST_QUERY_SET}.log"

    TEST_RESULT=$?

    # Save test artifacts
    sleep 10
    mkdir -p "${PROJECT_ROOT}/${TEST_RESULT_DIR}"

    [ -d logs ] && mv logs/* "${PROJECT_ROOT}/${TEST_RESULT_DIR}/" 2>/dev/null && echo "Saved test logs"

    cd "$PROJECT_ROOT"

    if [ $TEST_RESULT -ne 0 ]; then
        echo "ERROR: HybridQO testing failed!"
        exit 1
    fi

    TEST_HYBRIDQO_END=$(date +%s)
    TEST_HYBRIDQO_TIME=$((TEST_HYBRIDQO_END - TEST_HYBRIDQO_START))
    echo "HybridQO testing completed in ${TEST_HYBRIDQO_TIME}s"
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
echo "  Training:   Dataset=$TRAIN_DATASET, QuerySet=$TRAIN_QUERY_SET, Epochs=$EPOCH"
echo "  Testing:    Dataset=$TEST_DATASET, QuerySet=$TEST_QUERY_SET"
if [ "$SKIP_TRAINING" = true ]; then
echo "  Model:      Loaded from cache"
echo "              $MODEL_SAVE_DIR"
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
if [ "$SKIP_HYBRIDQO_TEST" = true ]; then
echo "  HybridQO Testing:   (cached)"
else
echo "  HybridQO Testing:   ${TEST_HYBRIDQO_TIME}s ($(($TEST_HYBRIDQO_TIME / 60))m)"
fi
echo "  Total:              ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Files in: $LOG_DIR/"
echo "  Model:        ${MODEL_SAVE_DIR}"
if [ "$SKIP_HYBRIDQO_TEST" = true ]; then
echo "  HybridQO Test: $EXISTING_HYBRIDQO_TEST (cached)"
else
echo "  HybridQO Test: ${TODAY}_test_hybridqo_${TEST_DATASET}_${TEST_QUERY_SET}.log"
fi
echo ""
echo "End Time: $(date)"
echo "================================================================"
