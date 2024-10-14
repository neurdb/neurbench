#!/usr/bin/env bash

set -x
set -e

DATASETS_DIR=/home/haotian/r/neurdb/AI4QueryOptimizer/datasets

DBNAME=job
TABLE_NAMES=(
    aka_name
    aka_title
    cast_info
    char_name
    comp_cast_type
    company_name
    company_type
    complete_cast
    info_type
    keyword
    kind_type
    link_type
    movie_companies
    movie_info
    movie_info_idx
    movie_keyword
    movie_link
    name
    person_info
    role_type
    title
)
N_BINS=20

# DRIFT_FACTOR=0.1
# DEST_DIR=data_1

# mkdir -p "${DATASETS_DIR}/imdb/${DEST_DIR}"

# for TABLE_NAME in "${TABLE_NAMES[@]}"; do
#     python dbproc.py -d "$DBNAME" -t "$TABLE_NAME" \
#         -i "${DATASETS_DIR}/imdb/imdb/${TABLE_NAME}.csv" \
#         -o "${DATASETS_DIR}/imdb/${DEST_DIR}/${TABLE_NAME}.csv" \
#         -b $N_BINS -D $DRIFT_FACTOR -s 1
#     head -n 10 "${DATASETS_DIR}/imdb/${DEST_DIR}/${TABLE_NAME}.csv" \
#         >"sample-${TABLE_NAME}-drifted-${DRIFT_FACTOR}.csv"
# done

DRIFT_FACTOR=0.3
DEST_DIR=data_2

mkdir -p "${DATASETS_DIR}/imdb/${DEST_DIR}"

for TABLE_NAME in "${TABLE_NAMES[@]}"; do
    python dbproc.py -d "$DBNAME" -t "$TABLE_NAME" \
        -i "${DATASETS_DIR}/imdb/imdb/${TABLE_NAME}.csv" \
        -o "${DATASETS_DIR}/imdb/${DEST_DIR}/${TABLE_NAME}.csv" \
        -b $N_BINS -D $DRIFT_FACTOR -s 1
    head -n 10 "${DATASETS_DIR}/imdb/${DEST_DIR}/${TABLE_NAME}.csv" \
        >"sample-${TABLE_NAME}-drifted-${DRIFT_FACTOR}.csv"
done

DRIFT_FACTOR=0.5
DEST_DIR=data_3

mkdir -p "${DATASETS_DIR}/imdb/${DEST_DIR}"

for TABLE_NAME in "${TABLE_NAMES[@]}"; do
    python dbproc.py -d "$DBNAME" -t "$TABLE_NAME" \
        -i "${DATASETS_DIR}/imdb/imdb/${TABLE_NAME}.csv" \
        -o "${DATASETS_DIR}/imdb/${DEST_DIR}/${TABLE_NAME}.csv" \
        -b $N_BINS -D $DRIFT_FACTOR -s 1
    head -n 10 "${DATASETS_DIR}/imdb/${DEST_DIR}/${TABLE_NAME}.csv" \
        >"sample-${TABLE_NAME}-drifted-${DRIFT_FACTOR}.csv"
done

DRIFT_FACTOR=0.7
DEST_DIR=data_4

mkdir -p "${DATASETS_DIR}/imdb/${DEST_DIR}"

for TABLE_NAME in "${TABLE_NAMES[@]}"; do
    python dbproc.py -d "$DBNAME" -t "$TABLE_NAME" \
        -i "${DATASETS_DIR}/imdb/imdb/${TABLE_NAME}.csv" \
        -o "${DATASETS_DIR}/imdb/${DEST_DIR}/${TABLE_NAME}.csv" \
        -b $N_BINS -D $DRIFT_FACTOR -s 1
    head -n 10 "${DATASETS_DIR}/imdb/${DEST_DIR}/${TABLE_NAME}.csv" \
        >"sample-${TABLE_NAME}-drifted-${DRIFT_FACTOR}.csv"
done
