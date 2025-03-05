#!/usr/bin/env bash

set -x

for drift in $(seq 0.1 0.01 0.7); do
    # scale=$(python -c "print(\"%.3f\" % (0.3 + ($drift - 0.1) * 0.5))") # no corr
    scale=$(python -c "print(\"%.3f\" % (0.5 + ($drift - 0.1) * 1.0))") # p1.0
    echo "drift=$drift, scale=$scale"
    python main.py --task-name=oversampling --dataset-name=default --device=7 --save-name=default_output \
        --diffuser-steps=30000 --diffuser-bs=2048 --controller-lr=0.0001 --controller-steps=10000 \
        --scale-factor=$scale --drift=$drift
    python postproc.py --expdir=expdir/default_output --enable-drift --enable-corr |
        grep -e 'mean absolute loss' -e 'mean JS divergence' |
        sed "s/^/$drift /" | tee -a loop_drift_results.txt
done
