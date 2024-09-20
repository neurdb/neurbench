#!/usr/bin/env bash

set -x

TABLE_NAME=customer
N_BINS=20
DRIFT_FACTOR=0.2

python dbproc.py -t $TABLE_NAME -i "tpch-kit/dbgen/1g/${TABLE_NAME}.tbl" -o "${TABLE_NAME}-drifted-1.tbl" -b $N_BINS -D $DRIFT_FACTOR -s 1
head -n 10 "${TABLE_NAME}-drifted-1.tbl" > "sample-${TABLE_NAME}-drifted-1.tbl"

for ((i = 1; i < 10; i++)); do
    drift_factor=$(echo "0.05 * $i" | bc)
    next=$((i + 1))

    if [ $((i % 2)) -eq 0 ]; then
        SKEWED=1
    else
        SKEWED=0
    fi

    python dbproc.py -t $TABLE_NAME -i "${TABLE_NAME}-drifted-${i}.tbl" -o "${TABLE_NAME}-drifted-${next}.tbl" -b $N_BINS -D $drift_factor -s $SKEWED
    head -n 10 "${TABLE_NAME}-drifted-${next}.tbl" >"sample-${TABLE_NAME}-drifted-${next}.tbl"
done
