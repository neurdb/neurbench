#!/bin/bash
#
# Check Bao Environment Setup
# Usage: ./check_bao_env.sh [QUERY_SET]
# This script verifies all prerequisites before running schedule_bao_simple.sh
#

echo "================================================================"
echo "Bao Environment Check"
echo "================================================================"

ERRORS=0
WARNINGS=0

# Configuration
DATASET="imdb_ori"
QUERY_SET="${1:-join-order-benchmark}"
PROJECT_ROOT="../../.."  # Path to project root from benchmarks/lqos/bao

echo "Dataset: $DATASET"
echo "Query Set: $QUERY_SET"
echo ""

# Check 1: Query directories
echo "[1] Checking query directories..."
echo "----------------------------------------------------------------"

TRAIN_DIR="$PROJECT_ROOT/queries/${QUERY_SET}/train"
TEST_DIR="$PROJECT_ROOT/queries/${QUERY_SET}/test"

if [ -d "$TRAIN_DIR" ]; then
    TRAIN_COUNT=$(find "$TRAIN_DIR" -name "*.sql" 2>/dev/null | wc -l)
    echo "✓ Training queries: $TRAIN_DIR ($TRAIN_COUNT files)"
else
    echo "✗ Training directory missing: $TRAIN_DIR"
    ERRORS=$((ERRORS + 1))
fi

if [ -d "$TEST_DIR" ]; then
    TEST_COUNT=$(find "$TEST_DIR" -name "*.sql" 2>/dev/null | wc -l)
    echo "✓ Test queries: $TEST_DIR ($TEST_COUNT files)"
else
    echo "✗ Test directory missing: $TEST_DIR"
    ERRORS=$((ERRORS + 1))
fi

# Check 2: Bao scripts
echo ""
echo "[2] Checking Bao scripts..."
echo "----------------------------------------------------------------"

TRAIN_SCRIPT="./train_bao.py"
TEST_SCRIPT="./test_bao.py"

if [ -f "$TRAIN_SCRIPT" ]; then
    echo "✓ Training script: $TRAIN_SCRIPT"
else
    echo "✗ Training script missing: $TRAIN_SCRIPT"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "$TEST_SCRIPT" ]; then
    echo "✓ Test script: $TEST_SCRIPT"
else
    echo "✗ Test script missing: $TEST_SCRIPT"
    ERRORS=$((ERRORS + 1))
fi

if [ -d "./bao_server" ]; then
    echo "✓ Bao server directory: ./bao_server"
else
    echo "✗ Bao server directory missing: ./bao_server"
    ERRORS=$((ERRORS + 1))
fi

# Check 3: Python dependencies
echo ""
echo "[3] Checking Python dependencies..."
echo "----------------------------------------------------------------"

python3 -c "import psycopg2; print('✓ psycopg2')" 2>/dev/null || { echo "✗ psycopg2 not installed"; ERRORS=$((ERRORS + 1)); }
python3 -c "import torch; print('✓ torch')" 2>/dev/null || { echo "✗ torch not installed"; ERRORS=$((ERRORS + 1)); }
python3 -c "import psutil; print('✓ psutil')" 2>/dev/null || { echo "✗ psutil not installed"; ERRORS=$((ERRORS + 1)); }

# Check 4: Database connection
echo ""
echo "[4] Checking database connection..."
echo "----------------------------------------------------------------"

DB_NAME="${DATASET}"
DB_HOST="172.17.0.1"
DB_USER="postgres"
DB_PASS="postgres"
DB_PORT="5430"

python3 << EOF
import psycopg2
try:
    conn = psycopg2.connect(
        dbname='$DB_NAME',
        user='$DB_USER',
        password='$DB_PASS',
        host='$DB_HOST',
        port='$DB_PORT',
        connect_timeout=3
    )
    print('✓ Database connection successful')
    print(f'  - Host: $DB_HOST:$DB_PORT')
    print(f'  - Database: $DB_NAME')
    conn.close()
except Exception as e:
    print(f'✗ Database connection failed: {e}')
    exit(1)
EOF

if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS + 1))
fi

# Check 5: CLI module
echo ""
echo "[5] Checking CLI module..."
echo "----------------------------------------------------------------"

python3 << EOF
try:
    import sys
    sys.path.insert(0, '$PROJECT_ROOT')
    from cli import GLOBAL_CONFIG, handle_tqo, handle_iqo
    print('✓ CLI module loaded successfully')
    print(f'  - Current dataset: {GLOBAL_CONFIG["dataset"]}')
    print(f'  - Current drift: {GLOBAL_CONFIG["drift"]}')
    print(f'  - PostgreSQL port: {GLOBAL_CONFIG["pg_port"]}')
except Exception as e:
    print(f'✗ Failed to load CLI module: {e}')
    exit(1)
EOF

if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS + 1))
fi

# Check 6: Log directory
echo ""
echo "[6] Checking log directory..."
echo "----------------------------------------------------------------"

LOG_DIR="$PROJECT_ROOT/bao_logs_all"
if [ -d "$LOG_DIR" ]; then
    echo "✓ Log directory exists: $LOG_DIR"
else
    echo "⚠ Log directory will be created: $LOG_DIR"
    WARNINGS=$((WARNINGS + 1))
fi

# Summary
echo ""
echo "================================================================"
echo "Summary"
echo "================================================================"

if [ $ERRORS -eq 0 ]; then
    echo "✓ All checks passed!"
    if [ $WARNINGS -gt 0 ]; then
        echo "⚠ $WARNINGS warning(s)"
    fi
    echo ""
    echo "You can now run:"
    echo "  ./schedule_bao_simple.sh $QUERY_SET"
    exit 0
else
    echo "✗ $ERRORS error(s) found"
    if [ $WARNINGS -gt 0 ]; then
        echo "⚠ $WARNINGS warning(s)"
    fi
    echo ""
    echo "Please fix the errors before running the pipeline."
    exit 1
fi
