#!/bin/bash
# for nh in 400 800; do  #10 25 50 100 200; do
#    for r in 0 5 10; do
# qsub -v MLP_R=$1,MLP_GPU=$2 run.sh
qsub -v MLP_GPU=$2 run.sh
#    done
# done