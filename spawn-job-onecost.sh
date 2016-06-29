#!/bin/bash:
GPU=0
for bs in 50; do  #150 250; do
   for dropout in 0.2 ; do # 0.5 0.1 0.0
        for cnl in "relu" "tanh"; do
        	for lr_decay in 0.9 0.95 1.0; do
        		for mode in add mul; do
        			GPU=$((GPU+1))
        			if [ $GPU -eq 3 ]; then
        				GPU=0
        			fi
		          	echo "Starting job with batch size: $bs, dropout: $dropout, conv_non_linear: $cnl lr_decay: $lr_decay in mode: $mode on GPU: $GPU"
		          	if [ $1 -eq 1 ]; then
		          		echo "Starting Model 2 with 3 CNNs"
						qsub -v BATCH_SIZE_F=$1,DROPOUT_F=$2,CNL_F=$3,GPU_NO=$4 job3.sh
					else
						echo "Starting Model 3 with MODE $mode"
						qsub -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh
					fi
				done
			done
		done
	done
done
