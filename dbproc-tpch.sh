#!/usr/bin/env bash

set -x
set -e

DATASETS_DIR=/home/haotian/r/neuralbench/neurbench/datasets/tpch/1g_csv
DBNAME=tpch
TABLE_NAMES=(
    customer
    lineitem
    nation
    orders
    part
    partsupp
    region
    supplier
)
N_BINS=20

DRIFT_FACTOR=0.1
DEST_DIR=data_01

mkdir -p "${DATASETS_DIR}/${DEST_DIR}"

for TABLE_NAME in "${TABLE_NAMES[@]}"; do
    python -u dbproc.py -d "$DBNAME" -t "$TABLE_NAME" \
        -i "${DATASETS_DIR}/${TABLE_NAME}.csv" \
        -o "${DATASETS_DIR}/${DEST_DIR}/${TABLE_NAME}.csv" \
        -b $N_BINS -D $DRIFT_FACTOR -s 1 > "${DATASETS_DIR}/${DEST_DIR}/${TABLE_NAME}.log" 2>&1
    head -n 10 "${DATASETS_DIR}/${DEST_DIR}/${TABLE_NAME}.csv" \
        >"sample-${TABLE_NAME}-drifted-${DRIFT_FACTOR}.csv"
done

DRIFT_FACTOR=0.3
DEST_DIR=data_03

mkdir -p "${DATASETS_DIR}/${DEST_DIR}"

for TABLE_NAME in "${TABLE_NAMES[@]}"; do
    python -u dbproc.py -d "$DBNAME" -t "$TABLE_NAME" \
        -i "${DATASETS_DIR}/${TABLE_NAME}.csv" \
        -o "${DATASETS_DIR}/${DEST_DIR}/${TABLE_NAME}.csv" \
        -b $N_BINS -D $DRIFT_FACTOR -s 1 > "${DATASETS_DIR}/${DEST_DIR}/${TABLE_NAME}.log" 2>&1
    head -n 10 "${DATASETS_DIR}/${DEST_DIR}/${TABLE_NAME}.csv" \
        >"sample-${TABLE_NAME}-drifted-${DRIFT_FACTOR}.csv"
done

DRIFT_FACTOR=0.5
DEST_DIR=data_05

mkdir -p "${DATASETS_DIR}/${DEST_DIR}"

for TABLE_NAME in "${TABLE_NAMES[@]}"; do
    python -u dbproc.py -d "$DBNAME" -t "$TABLE_NAME" \
        -i "${DATASETS_DIR}/${TABLE_NAME}.csv" \
        -o "${DATASETS_DIR}/${DEST_DIR}/${TABLE_NAME}.csv" \
        -b $N_BINS -D $DRIFT_FACTOR -s 1 > "${DATASETS_DIR}/${DEST_DIR}/${TABLE_NAME}.log" 2>&1
    head -n 10 "${DATASETS_DIR}/${DEST_DIR}/${TABLE_NAME}.csv" \
        >"sample-${TABLE_NAME}-drifted-${DRIFT_FACTOR}.csv"
done

DRIFT_FACTOR=0.7
DEST_DIR=data_07

mkdir -p "${DATASETS_DIR}/${DEST_DIR}"

for TABLE_NAME in "${TABLE_NAMES[@]}"; do
    python -u dbproc.py -d "$DBNAME" -t "$TABLE_NAME" \
        -i "${DATASETS_DIR}/${TABLE_NAME}.csv" \
        -o "${DATASETS_DIR}/${DEST_DIR}/${TABLE_NAME}.csv" \
        -b $N_BINS -D $DRIFT_FACTOR -s 1 > "${DATASETS_DIR}/${DEST_DIR}/${TABLE_NAME}.log" 2>&1
    head -n 10 "${DATASETS_DIR}/${DEST_DIR}/${TABLE_NAME}.csv" \
        >"sample-${TABLE_NAME}-drifted-${DRIFT_FACTOR}.csv"
done
