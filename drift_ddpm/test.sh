#!/usr/bin/env bash

set -x

#for drift in $(seq 0.1 0.01 0.7); do
# scale=$(python -c "print(\"%.3f\" % (0.3 + ($drift - 0.1) * 0.5))") # no corr
drift=0.1
scale=$(python -c "print(\"%.3f\" % (0.5 + ($drift - 0.1) * 1.0))") # p1.0
echo "drift=$drift, scale=$scale"
python gen.py --dataset-name=imdb --table-name=movie_link  --drift=$drift  --device=7 \
        --diffuser-steps=30000 --diffuser-bs=2048 --controller-lr=0.0001 --controller-steps=10000 \
        --scale-factor=$scale
python postproc.py --dataset-name=imdb --table-name=movie_link --enable-drift --enable-corr |
grep -e 'mean absolute loss' -e 'mean JS divergence' |
sed "s/^/$drift /" | tee -a loop_drift_results.txt
#done
