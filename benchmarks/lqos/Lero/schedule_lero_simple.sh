#!/bin/bash
#
# Lero Training and Testing Pipeline
# Usage: ./schedule_lero_simple.sh [QUERY_SET]
#        ./schedule_lero_simple.sh [OPTIONS]
#
# Simple mode (backward compatible):
#   ./schedule_lero_simple.sh                           # Use default query_set
#   ./schedule_lero_simple.sh job      # Use specified query_set
#
# Advanced mode (separate train/test configuration):
#   ./schedule_lero_simple.sh --train-dataset imdb --train-query-set job-train \
#                             --test-dataset imdb_drift --test-query-set job-test
#
# Options:
#   --train-dataset DATASET      Dataset for training (default: imdb)
#   --train-query-set QUERY_SET  Query set for training (default: job)
#   --test-dataset DATASET       Dataset for testing (default: same as train-dataset)
#   --test-query-set QUERY_SET   Query set for testing (default: same as train-query-set)
#   --force-retrain              Force retraining even if model exists
#   --force-retest               Force retesting even if test results exist
#   --training-style STYLE       Training style: 'lero' (cardinality-guided, default) or 'bao' (hint-based)
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
    echo "Lero Training and Testing Pipeline"
    echo ""
    echo "Usage:"
    echo "  ./schedule_lero_simple.sh [QUERY_SET]"
    echo "  ./schedule_lero_simple.sh [OPTIONS]"
    echo ""
    echo "Simple mode (backward compatible):"
    echo "  ./schedule_lero_simple.sh                      # Use defaults"
    echo "  ./schedule_lero_simple.sh job # Use specified query_set"
    echo ""
    echo "Options:"
    echo "  --train-dataset DATASET      Dataset for training (default: imdb)"
    echo "  --train-query-set QUERY_SET  Query set for training (default: job)"
    echo "  --test-dataset DATASET       Dataset for testing (default: same as train-dataset)"
    echo "  --test-query-set QUERY_SET   Query set for testing (default: same as train-query-set)"
    echo "  --force-retrain              Force retraining even if model exists"
    echo "  --force-retest               Force retesting even if test results exist"
    echo "  --training-style STYLE       Training style: 'lero' (default) or 'bao'"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Note: The script auto-detects existing models and test results."
    echo "      - Models: loaded from cache if train-dataset/query-set matches"
    echo "      - Tests: skipped if test-dataset/query-set results exist"
    echo "      Use --force-retrain or --force-retest to override."
    echo ""
    echo "Examples:"
    echo "  # Same dataset/query for train and test"
    echo "  ./schedule_lero_simple.sh job"
    echo ""
    echo "  # Different datasets for train and test"
    echo "  ./schedule_lero_simple.sh --train-dataset imdb --test-dataset imdb_drift"
    echo ""
    echo "  # Fully customized"
    echo "  ./schedule_lero_simple.sh --train-dataset imdb --train-query-set job-train \\"
    echo "                            --test-dataset imdb_drift --test-query-set job-test"
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

# Lero directories (use absolute paths)
LERO_DIR="$PROJECT_ROOT/benchmarks/lqos/Lero"
LERO_CORE_DIR="$LERO_DIR/lero"
LERO_TEST_SCRIPT_DIR="$LERO_CORE_DIR/test_script"
CONFIG_FILE="$LERO_TEST_SCRIPT_DIR/config.py"

# Verify Lero exists
if [ ! -d "$LERO_DIR" ]; then
    echo "ERROR: Lero directory not found at $LERO_DIR"
    exit 1
fi

# Default configuration
TRAIN_DATASET="imdb"
TRAIN_QUERY_SET="job"
TEST_DATASET=""
TEST_QUERY_SET=""
FORCE_RETRAIN=false
FORCE_RETEST=false
TRAINING_STYLE="lero"  # "lero" (cardinality-guided) or "bao" (hint-based)

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
            --training-style)
                TRAINING_STYLE="$2"
                shift 2
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
LOG_DIR="./lero_logs_all"
TODAY=$(date +%Y%m%d)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PG_PORT=5434

echo "================================================================"
echo "Lero Training and Testing Pipeline"
echo "Training:"
echo "  Dataset:   $TRAIN_DATASET"
echo "  Query Set: $TRAIN_QUERY_SET"
echo "  Style:     $TRAINING_STYLE"
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

# Function to convert SQL files to Lero format (query_name#####SQL)
prepare_query_file() {
    local sql_dir="$1"
    local output_file="$2"

    echo "Preparing query file from $sql_dir..."

    if [ ! -d "$sql_dir" ]; then
        echo "ERROR: Query directory not found: $sql_dir"
        return 1
    fi

    # Clear output file
    > "$output_file"

    # Convert each SQL file to Lero format
    for sql_file in "$sql_dir"/*.sql; do
        if [ -f "$sql_file" ]; then
            query_name=$(basename "$sql_file" .sql)
            # Read SQL content and remove newlines
            sql_content=$(cat "$sql_file" | tr '\n' ' ' | tr -s ' ')
            echo "${query_name}#####${sql_content}" >> "$output_file"
        fi
    done

    local count=$(wc -l < "$output_file")
    echo "Prepared $count queries in $output_file"
    return 0
}

# Function to update config.py with database name
update_config() {
    local db_name="$1"
    echo "Updating config.py with DB=$db_name..."
    sed -i "s|^DB = .*|DB = \"$db_name\"|" "$CONFIG_FILE"
}

# Function to kill Lero server
kill_lero_server() {
    echo "Stopping Lero server..."
    pkill -f "python.*server.py" || true
    pkill -f "python.*server.py" || true
    sleep 5
}

# Timing
START_TIME=$(date +%s)

# ============================================================
# Step 1: Train Lero (or load existing model)
# ============================================================
echo ""
echo "[1/2] Training Lero with $TRAIN_DATASET..."
echo "----------------------------------------------------------------"

TRAIN_TIME_START=$(date +%s)
SKIP_TRAINING=false

# Model pattern for Lero
MODEL_PREFIX="train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_model"
EXISTING_MODEL=$(ls -dt ${LERO_CORE_DIR}/${MODEL_PREFIX}_* 2>/dev/null | head -1)

if [ "$FORCE_RETRAIN" = true ]; then
    echo "--force-retrain specified, will retrain model"
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

    # Prepare training query file
    TRAIN_SQL_DIR="queries/${TRAIN_QUERY_SET}"
    TRAIN_QUERY_FILE="${LOG_DIR}/${TRAIN_DATASET}_${TRAIN_QUERY_SET}_train.txt"

    if ! prepare_query_file "$TRAIN_SQL_DIR" "$TRAIN_QUERY_FILE"; then
        echo "ERROR: Failed to prepare training queries"
        exit 1
    fi

    # Update config with training database
    update_config "$TRAIN_DATASET"

    # Kill any existing Lero server
    kill_lero_server

    # Start Lero server
    echo "Starting Lero server..."
    cd "$LERO_CORE_DIR"
    nohup env CUDA_VISIBLE_DEVICES="" python3 -u server.py >> "${PROJECT_ROOT}/${LOG_DIR}/${TODAY}_server_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log" 2>&1 &
    LERO_SERVER_PID=$!
    echo "Lero server started with PID: $LERO_SERVER_PID"
    sleep 15

    # Run training
    echo "Running Lero training..."
    cd "$LERO_CORE_DIR/test_script"

    CUDA_VISIBLE_DEVICES="" python3 train_model.py \
        --query_path "${PROJECT_ROOT}/${TRAIN_QUERY_FILE}" \
        --test_query_path "${PROJECT_ROOT}/${TRAIN_QUERY_FILE}" \
        --algo lero \
        --query_num_per_chunk 20 \
        --output_query_latency_file "${PROJECT_ROOT}/${LOG_DIR}/${TODAY}_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log" \
        --model_prefix "${MODEL_PREFIX}" \
        --topK 3 \
        --training_style "${TRAINING_STYLE}" \
        2>&1 | tee "${PROJECT_ROOT}/${LOG_DIR}/${TODAY}_train_output_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log"

    TRAIN_RESULT=$?

    # Save training log directory
    if [ -d "${LERO_CORE_DIR}/test_script/log" ]; then
        TRAIN_LOG_SAVE_DIR="${PROJECT_ROOT}/${LOG_DIR}/log_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}"
        mv "${LERO_CORE_DIR}/test_script/log" "$TRAIN_LOG_SAVE_DIR" 2>/dev/null || true
        echo "Saved training logs to: $TRAIN_LOG_SAVE_DIR"
    fi

    cd "$PROJECT_ROOT"

    # Stop server after training
    kill_lero_server

    if [ $TRAIN_RESULT -ne 0 ]; then
        echo "ERROR: Training failed!"
        exit 1
    fi

    TRAIN_TIME_END=$(date +%s)
    TRAIN_TIME=$((TRAIN_TIME_END - TRAIN_TIME_START))
    echo "Training completed in ${TRAIN_TIME}s"

    # Find the latest model
    EXISTING_MODEL=$(ls -dt ${LERO_CORE_DIR}/${MODEL_PREFIX}_* 2>/dev/null | head -1)
fi

# Ensure Lero server is stopped after training
kill_lero_server

# ============================================================
# Step 2: Test with Lero
# ============================================================
echo ""
echo "[2/2] Testing with Lero optimizer..."
echo "----------------------------------------------------------------"

TEST_LERO_START=$(date +%s)
SKIP_LERO_TEST=false

# Check if Lero test results already exist
EXISTING_LERO_TEST=$(ls -t ${LOG_DIR}/*_test_lero_${TEST_DATASET}_${TEST_QUERY_SET}.log 2>/dev/null | head -1)

if [ "$FORCE_RETEST" = true ]; then
    echo "--force-retest specified, will run Lero test"
    EXISTING_LERO_TEST=""
fi

if [ -n "$EXISTING_LERO_TEST" ] && [ -s "$EXISTING_LERO_TEST" ]; then
    echo "Found existing Lero test results: $EXISTING_LERO_TEST"
    echo "  Skipping Lero test (use --force-retest to override)"
    SKIP_LERO_TEST=true
    TEST_LERO_TIME=0
else
    # Prepare test query file
    TEST_SQL_DIR="queries/${TEST_QUERY_SET}"
    TEST_QUERY_FILE="${LOG_DIR}/${TEST_DATASET}_${TEST_QUERY_SET}_test.txt"

    if ! prepare_query_file "$TEST_SQL_DIR" "$TEST_QUERY_FILE"; then
        echo "ERROR: Failed to prepare test queries"
        exit 1
    fi

    # Update config with test database
    update_config "$TEST_DATASET"

    # Kill any existing Lero server
    kill_lero_server

    # Get model number (find the highest numbered model)
    MODEL_NUM=$(ls -d ${LERO_CORE_DIR}/${MODEL_PREFIX}_* 2>/dev/null | sed 's/.*_//' | sort -n | tail -1)
    if [ -z "$MODEL_NUM" ]; then
        echo "ERROR: No trained model found for ${MODEL_PREFIX}"
        exit 1
    fi
    FULL_MODEL_PATH="${MODEL_PREFIX}_${MODEL_NUM}"
    echo "Using model: $FULL_MODEL_PATH"

    # Update server_test.conf with model path (needs [lero] section header)
    echo "[lero]" > "${LERO_CORE_DIR}/server_test.conf"
    echo "Port = 14567" >> "${LERO_CORE_DIR}/server_test.conf"
    echo "ListenOn = localhost" >> "${LERO_CORE_DIR}/server_test.conf"
    echo "ModelPath = ${FULL_MODEL_PATH}" >> "${LERO_CORE_DIR}/server_test.conf"

    # Start Lero server with model
    echo "Starting Lero server with model..."
    cd "$LERO_CORE_DIR"
    nohup env CUDA_VISIBLE_DEVICES="" python3 -u server.py --config_name server_test.conf >> "${PROJECT_ROOT}/${LOG_DIR}/${TODAY}_server_test_lero_${TEST_DATASET}_${TEST_QUERY_SET}.log" 2>&1 &
    sleep 15

    # Run test
    echo "Running Lero test..."
    cd "$LERO_CORE_DIR/test_script"

    # Clean up previous log directory before test
    rm -rf log 2>/dev/null || true

    python3 test.py \
        --query_path "${PROJECT_ROOT}/${TEST_QUERY_FILE}" \
        --output_query_latency_file "${PROJECT_ROOT}/${LOG_DIR}/${TODAY}_test_lero_${TEST_DATASET}_${TEST_QUERY_SET}.log" \
        2>&1 | tee "${PROJECT_ROOT}/${LOG_DIR}/${TODAY}_test_lero_output_${TEST_DATASET}_${TEST_QUERY_SET}.log"

    TEST_RESULT=$?

    # Save test log directory
    if [ -d "log" ]; then
        TEST_LOG_SAVE_DIR="${PROJECT_ROOT}/${LOG_DIR}/log_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_test_${TEST_DATASET}_${TEST_QUERY_SET}"
        mv log "$TEST_LOG_SAVE_DIR" 2>/dev/null || true
        echo "Saved test logs to: $TEST_LOG_SAVE_DIR"
    fi

    cd "$PROJECT_ROOT"

    # Stop server
    kill_lero_server

    if [ $TEST_RESULT -ne 0 ]; then
        echo "ERROR: Lero testing failed!"
        exit 1
    fi

    TEST_LERO_END=$(date +%s)
    TEST_LERO_TIME=$((TEST_LERO_END - TEST_LERO_START))
    echo "Lero testing completed in ${TEST_LERO_TIME}s"
fi

# Final cleanup
kill_lero_server

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
echo "  Training:   Dataset=$TRAIN_DATASET, QuerySet=$TRAIN_QUERY_SET, Style=$TRAINING_STYLE"
echo "  Testing:    Dataset=$TEST_DATASET, QuerySet=$TEST_QUERY_SET"
if [ "$SKIP_TRAINING" = true ]; then
echo "  Model:      Loaded from cache"
echo "              $EXISTING_MODEL"
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
if [ "$SKIP_LERO_TEST" = true ]; then
echo "  Lero Testing:       (cached)"
else
echo "  Lero Testing:       ${TEST_LERO_TIME}s ($(($TEST_LERO_TIME / 60))m)"
fi
echo "  Total:              ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Files in: $LOG_DIR/"
echo "  Model:     ${LERO_CORE_DIR}/${MODEL_PREFIX}_*"
if [ "$SKIP_LERO_TEST" = true ]; then
echo "  Lero Test: $EXISTING_LERO_TEST (cached)"
else
echo "  Lero Test: ${TODAY}_test_lero_${TEST_DATASET}_${TEST_QUERY_SET}.log"
fi
echo ""
echo "End Time: $(date)"
echo "================================================================"
