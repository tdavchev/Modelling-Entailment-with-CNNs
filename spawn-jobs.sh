#!/bin/bash
 #for bs in 100 150 200; do  #10 25 50 100 200; do
 #   for dropout in 0.5 0.2 0.0; do
# qsub -v MLP_R=$1,MLP_GPU=$2 run.sh
#       for cnl in "relu" "tanh"; do
#          echo "Starting job with batch size: $bs, dropout: $dropout, conv_non_linear: $cnl"
echo "$1,$2,$3"
qsub -v BATCH_SIZE=$1,DROPOUT=$2,CNL=$3,GPU_NO=$4 job.sh
#       done       
#    done
# done
