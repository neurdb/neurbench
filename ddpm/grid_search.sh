#!/usr/bin/env bash

set -x

DEVICE=1

for lambda_p in $(seq 0 0.1 1.0); do
    for lambda_s in $(seq 0 0.1 1.0); do
        python main.py --task-name=oversampling --dataset-name=default --device=$DEVICE \
            --lambda-p $lambda_p --lambda-s $lambda_s \
            --save-name=default_output-${lambda_p}-${lambda_s} \
            --diffuser-steps=30000 --diffuser-bs=2048 --retrain-diffuser

        python postproc.py --expdir=expdir/default_output-${lambda_p}-${lambda_s} |\
            grep 'mean absolute loss' |\
            sed "s/^/$lambda_p $lambda_s /" >> grid_search_results.txt
    done
done
