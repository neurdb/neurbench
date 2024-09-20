#!/usr/bin/env bash

set -x

N_BINS=20
DRIFT_FACTOR=0.2

python dbproc.py -t customer -i tpch-kit/dbgen/1g/customer.tbl -o customer-drifted-1.tbl -b $N_BINS -D $DRIFT_FACTOR -s 1
head -n 10 customer-drifted-1.tbl > customer-drifted-1-sample.tbl


for((i=1;i<10;i++)); do
    drift_factor=$(echo "0.05 * $i" | bc)
    next=$((i+1))

    if [ $((i % 2)) -eq 0 ]; then
        SKEWED=1
    else
        SKEWED=0
    fi

    python dbproc.py -t customer -i customer-drifted-$i.tbl -o "customer-drifted-$next.tbl" -b $N_BINS -D $drift_factor -s $SKEWED
    head -n 10 "customer-drifted-$next.tbl" > "customer-drifted-$next-sample.tbl"
done
