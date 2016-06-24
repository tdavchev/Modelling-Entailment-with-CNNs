#!/bin/bash
 #for bs in 100 150 200; do  #10 25 50 100 200; do
 #   for dropout in 0.5 0.2 0.0; do
# qsub -v MLP_R=$1,MLP_GPU=$2 run.sh
#       for cnl in "relu" "tanh"; do
#          echo "Starting job with batch size: $bs, dropout: $dropout, conv_non_linear: $cnl"
echo "$1,$2,$3"
if [ $1 -eq 1 ]; then
	echo "Starting Model 1"
	qsub -v BATCH_SIZE=$1,DROPOUT=$2,CNL=$3,GPU_NO=$4,MODEL=$5 job.sh
else 
	echo "Starting Model 2"
        qsub -v BATCH_SIZE_F=$1,DROPOUT_F=$2,CNL_F=$3,GPU_NO=$4,MODEL=$5,BATCH_SIZE_S=$6,DROPOUT_S=$7,CNL_S=$8 job.sh
fi

#       done       
#    done
# done
