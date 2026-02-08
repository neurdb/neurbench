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
set -o pipefail  # Catch errors in pipelines (e.g., cmd | tee)

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
MIN_QUERIES=""  # Minimum training queries (will sample with replacement if needed)
CONTINUE_TRAINING=false  # Continue training from existing model and data
TEST_INTERVAL=""  # Run test every N iterations (default: 25)

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
            --min-queries)
                MIN_QUERIES="$2"
                shift 2
                ;;
            --continue-training)
                CONTINUE_TRAINING=true
                shift
                ;;
            --test-interval)
                TEST_INTERVAL="$2"
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

    # Skip if output file already exists
    if [ -f "$output_file" ]; then
        local count=$(wc -l < "$output_file")
        echo "Query file already exists: $output_file ($count queries), skipping generation"
        return 0
    fi

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

# Search for any model matching dataset/query_set pattern (both old and new naming formats)
# Old format: train_{dataset}_{queryset}_model_*
# New format: {timestamp}_train_{dataset}_{queryset}_model_*
EXISTING_MODEL=$(ls -dt ${LERO_CORE_DIR}/*train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_model_* 2>/dev/null | head -1 || true)

if [ "$FORCE_RETRAIN" = true ]; then
    echo "--force-retrain specified, will retrain model"
    EXISTING_MODEL=""
fi

# Model pattern for Lero
# When continuing training, extract the prefix from existing model to match chunk numbering
# Otherwise, use new timestamp to avoid overwriting previous models
if [ "$CONTINUE_TRAINING" = true ] && [ -n "$EXISTING_MODEL" ]; then
    # Extract prefix from existing model: e.g., "/path/to/20260118_051201_train_imdb_job_model_32" -> "20260118_051201_train_imdb_job_model"
    EXISTING_MODEL_BASENAME=$(basename "$EXISTING_MODEL")
    # Remove trailing _NUMBER (chunk number) from basename
    MODEL_PREFIX=$(echo "$EXISTING_MODEL_BASENAME" | sed 's/_[0-9]*$//')
    echo "Continue training mode: using existing model prefix: $MODEL_PREFIX"
else
    MODEL_PREFIX="${TIMESTAMP}_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_model"
fi

if [ -n "$EXISTING_MODEL" ] && [ "$CONTINUE_TRAINING" = false ]; then
    echo "Found existing model: $EXISTING_MODEL"
    echo "  Skipping training (use --force-retrain to override or --continue-training to continue)"
    SKIP_TRAINING=true
    TRAIN_TIME=0
else
    # Continue training mode: find existing data and model
    CONTINUE_DATA_FILE=""
    CONTINUE_MODEL=""
    if [ "$CONTINUE_TRAINING" = true ]; then
        echo "Continue training mode enabled"
        # Find existing training data file (exclude server logs and output logs)
        # Pattern: {timestamp}_train_{dataset}_{queryset}.log (NOT server_train or train_output)
        EXISTING_DATA=$(ls -t ${LOG_DIR}/*_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log 2>/dev/null | grep -v '_server_train_' | grep -v '_train_output_' | head -1 || true)
        if [ -n "$EXISTING_DATA" ]; then
            CONTINUE_DATA_FILE="$EXISTING_DATA"
            echo "Found existing training data: $CONTINUE_DATA_FILE"
        fi
        # Find existing model to continue from
        if [ -n "$EXISTING_MODEL" ]; then
            CONTINUE_MODEL="$EXISTING_MODEL"
            echo "Will continue training from model: $CONTINUE_MODEL"
        fi
    fi

    if [ -n "$EXISTING_MODEL" ] && [ "$CONTINUE_TRAINING" = true ]; then
        echo "Continuing training from existing model: $EXISTING_MODEL"
    else
        echo "No existing model found for pattern *train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_model_*"
    fi
    echo "Starting training with model prefix: ${MODEL_PREFIX}"

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

    # Start Lero server (load existing model if continuing)
    echo "Starting Lero server..."
    cd "$LERO_CORE_DIR"
    if [ -n "$CONTINUE_MODEL" ]; then
        # Create a temporary config with model path for continue training
        CONTINUE_SERVER_CONF="${LERO_CORE_DIR}/server_continue.conf"
        CONTINUE_MODEL_BASENAME=$(basename "$CONTINUE_MODEL")
        cat > "$CONTINUE_SERVER_CONF" << EOF
[lero]
Port = 14567
ListenOn = 0.0.0.0
ModelPath = ${CONTINUE_MODEL_BASENAME}
EOF
        nohup env CUDA_VISIBLE_DEVICES="4" python3 -u server.py --config_name server_continue.conf >> "${PROJECT_ROOT}/${LOG_DIR}/${TIMESTAMP}_server_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log" 2>&1 &
    else
        nohup env CUDA_VISIBLE_DEVICES="4" python3 -u server.py >> "${PROJECT_ROOT}/${LOG_DIR}/${TIMESTAMP}_server_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log" 2>&1 &
    fi
    LERO_SERVER_PID=$!
    echo "Lero server started with PID: $LERO_SERVER_PID"
    sleep 15

    # Prewarm database once before training
    echo "Prewarming database before training..."
    cd "$LERO_CORE_DIR/test_script"
    python3 -c "from utils import prewarm_database; prewarm_database()"

    # Run training
    echo "Running Lero training..."
    export LERO_SKIP_PREWARM=1  # Skip per-query prewarm since we did it once above

    # Build min_queries argument if specified
    MIN_QUERIES_ARG=""
    if [ -n "$MIN_QUERIES" ]; then
        MIN_QUERIES_ARG="--min_queries $MIN_QUERIES"
        echo "Using min_queries: $MIN_QUERIES"
    fi

    # Build test_interval argument if specified
    TEST_INTERVAL_ARG=""
    if [ -n "$TEST_INTERVAL" ]; then
        TEST_INTERVAL_ARG="--test_interval $TEST_INTERVAL"
        echo "Using test_interval: $TEST_INTERVAL"
    fi

    # Build continue training arguments
    CONTINUE_ARGS=""
    if [ "$CONTINUE_TRAINING" = true ]; then
        CONTINUE_ARGS="$CONTINUE_ARGS --resume"  # Resume from last completed chunk
        if [ -n "$CONTINUE_DATA_FILE" ]; then
            CONTINUE_ARGS="$CONTINUE_ARGS --continue_data_file ${PROJECT_ROOT}/${CONTINUE_DATA_FILE}"
        fi
        if [ -n "$CONTINUE_MODEL" ]; then
            CONTINUE_ARGS="$CONTINUE_ARGS --continue_model ${CONTINUE_MODEL}"
        fi
    fi

    # Determine output file: reuse existing if continuing, otherwise new timestamp
    if [ "$CONTINUE_TRAINING" = true ] && [ -n "$CONTINUE_DATA_FILE" ]; then
        OUTPUT_LATENCY_FILE="${PROJECT_ROOT}/${CONTINUE_DATA_FILE}"
        echo "Appending to existing data file: $OUTPUT_LATENCY_FILE"
    else
        OUTPUT_LATENCY_FILE="${PROJECT_ROOT}/${LOG_DIR}/${TIMESTAMP}_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log"
    fi

    CUDA_VISIBLE_DEVICES="4,5,6" python3 train_model.py \
        --query_path "${PROJECT_ROOT}/${TRAIN_QUERY_FILE}" \
        --test_query_path "${PROJECT_ROOT}/${TRAIN_QUERY_FILE}" \
        --algo lero \
        --query_num_per_chunk 20 \
        --output_query_latency_file "$OUTPUT_LATENCY_FILE" \
        --model_prefix "${MODEL_PREFIX}" \
        --topK 3 \
        --training_style "${TRAINING_STYLE}" \
        $MIN_QUERIES_ARG \
        $TEST_INTERVAL_ARG \
        $CONTINUE_ARGS \
        2>&1 | tee "${PROJECT_ROOT}/${LOG_DIR}/${TIMESTAMP}_train_output_${TRAIN_DATASET}_${TRAIN_QUERY_SET}.log"

    TRAIN_RESULT=$?

    # Save training log directory (with timestamp to avoid overwriting)
    if [ -d "${LERO_CORE_DIR}/test_script/log" ]; then
        TRAIN_LOG_SAVE_DIR="${PROJECT_ROOT}/${LOG_DIR}/${TIMESTAMP}_log_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}"
        # Remove target if exists (shouldn't happen with timestamp, but be safe)
        rm -rf "$TRAIN_LOG_SAVE_DIR" 2>/dev/null || true
        mv "${LERO_CORE_DIR}/test_script/log" "$TRAIN_LOG_SAVE_DIR"
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

    # Find the latest model (may be from this run or previous runs)
    EXISTING_MODEL=$(ls -dt ${LERO_CORE_DIR}/*train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_model_* 2>/dev/null | head -1 || true)
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
EXISTING_LERO_TEST=$(ls -t ${LOG_DIR}/*_test_lero_${TEST_DATASET}_${TEST_QUERY_SET}.log 2>/dev/null | head -1 || true)

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

    # Try to find best model from checkpoint directory first
    # Look for best_model file in timestamped subdirectories matching the model prefix pattern
    # Path: lero_checkpoints/{timestamp}_{model_prefix}/best_model_{model_prefix}.txt
    BEST_MODEL_PATTERN="${PROJECT_ROOT}/${LOG_DIR}/lero_checkpoints/*_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_model/best_model_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_model.txt"
    BEST_MODEL_FILE=$(ls -t $BEST_MODEL_PATTERN 2>/dev/null | head -1 || true)
    BEST_MODEL=""
    if [ -n "$BEST_MODEL_FILE" ] && [ -f "$BEST_MODEL_FILE" ]; then
        BEST_MODEL=$(cat "$BEST_MODEL_FILE")
        if [ -d "${LERO_CORE_DIR}/${BEST_MODEL}" ]; then
            echo "Found best model from early stopping: $BEST_MODEL"
            echo "  (from $BEST_MODEL_FILE)"
            FULL_MODEL_PATH="$BEST_MODEL"
        else
            echo "Best model file exists but model directory not found: ${LERO_CORE_DIR}/${BEST_MODEL}"
            BEST_MODEL=""
        fi
    fi

    # If no best model, get the latest model directory (supports both old and new naming formats)
    if [ -z "$BEST_MODEL" ]; then
        LATEST_MODEL=$(ls -dt ${LERO_CORE_DIR}/*train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_model_* 2>/dev/null | head -1 || true)
        if [ -z "$LATEST_MODEL" ]; then
            echo "ERROR: No trained model found for pattern *train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_model_*"
            exit 1
        fi
        FULL_MODEL_PATH=$(basename "$LATEST_MODEL")
        echo "Using latest model: $FULL_MODEL_PATH"
    else
        echo "Using best model: $FULL_MODEL_PATH"
    fi

    # Update server_test.conf with model path (needs [lero] section header)
    echo "[lero]" > "${LERO_CORE_DIR}/server_test.conf"
    echo "Port = 14567" >> "${LERO_CORE_DIR}/server_test.conf"
    echo "ListenOn = 0.0.0.0" >> "${LERO_CORE_DIR}/server_test.conf"
    echo "ModelPath = ${FULL_MODEL_PATH}" >> "${LERO_CORE_DIR}/server_test.conf"

    # Start Lero server with model
    echo "Starting Lero server with model..."
    cd "$LERO_CORE_DIR"
    nohup env CUDA_VISIBLE_DEVICES="" python3 -u server.py --config_name server_test.conf >> "${PROJECT_ROOT}/${LOG_DIR}/${TIMESTAMP}_server_test_lero_${TEST_DATASET}_${TEST_QUERY_SET}.log" 2>&1 &
    sleep 15

    # Prewarm database once before testing
    echo "Prewarming database before testing..."
    cd "$LERO_CORE_DIR/test_script"
    python3 -c "from utils import prewarm_database; prewarm_database()"

    # Run test
    echo "Running Lero test..."
    export LERO_SKIP_PREWARM=1  # Skip per-query prewarm since we did it once above

    # Clean up previous log directory before test
    rm -rf log 2>/dev/null || true

    python3 test.py \
        --query_path "${PROJECT_ROOT}/${TEST_QUERY_FILE}" \
        --output_query_latency_file "${PROJECT_ROOT}/${LOG_DIR}/${TIMESTAMP}_test_lero_${TEST_DATASET}_${TEST_QUERY_SET}.log" \
        2>&1 | tee "${PROJECT_ROOT}/${LOG_DIR}/${TIMESTAMP}_test_lero_output_${TEST_DATASET}_${TEST_QUERY_SET}.log"

    TEST_RESULT=$?

    # Save test log directory (with timestamp to avoid overwriting)
    if [ -d "log" ]; then
        TEST_LOG_SAVE_DIR="${PROJECT_ROOT}/${LOG_DIR}/${TIMESTAMP}_log_train_${TRAIN_DATASET}_${TRAIN_QUERY_SET}_test_${TEST_DATASET}_${TEST_QUERY_SET}"
        # Remove target if exists (shouldn't happen with timestamp, but be safe)
        rm -rf "$TEST_LOG_SAVE_DIR" 2>/dev/null || true
        mv log "$TEST_LOG_SAVE_DIR"
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
SECS=$((TOTAL_TIME % 60))

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
echo "  Total:              ${HOURS}h ${MINUTES}m ${SECS}s"
echo ""
echo "Files in: $LOG_DIR/"
if [ "$SKIP_TRAINING" = true ]; then
echo "  Model:     $EXISTING_MODEL"
else
echo "  Model:     ${LERO_CORE_DIR}/${MODEL_PREFIX}_*"
fi
if [ "$SKIP_LERO_TEST" = true ]; then
echo "  Lero Test: $EXISTING_LERO_TEST (cached)"
else
echo "  Lero Test: ${TIMESTAMP}_test_lero_${TEST_DATASET}_${TEST_QUERY_SET}.log"
fi
echo ""
echo "End Time: $(date)"
echo "================================================================"
