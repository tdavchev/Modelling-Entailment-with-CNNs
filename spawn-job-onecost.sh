#!/bin/bash:
GPU=0
for bs in 150 200 250; do  #10 25 50 100 200; do
   for dropout in 0.5 0.2 0.1 0.0; do
        for cnl in "relu" "tanh"; do
        	for lr_decay in 0.9 0.95 1; do
        		for mode in add mul; do
        			GPU=$((GPU+1))
        			if [ $GPU -eq 2 ]; then
        				GPU=0
        			fi
		          	echo "Starting job with batch size: $bs, dropout: $dropout, conv_non_linear: $cnl"
		          	if [ $1 -eq 1 ]; then
		          		echo "Starting Model 2 with 3 CNNs"
						qsub -v BATCH_SIZE_F=$1,DROPOUT_F=$2,CNL_F=$3,GPU_NO=$4 job3.sh
					else
						echo "Starting Model 3 with MODE $6"
						qsub -v BATCH_SIZE_F=bs,DROPOUT_F=dropout,CNL_F=cnl,GPU_NO=GPU,MODE=mode,LR_DECAY=lr_decay job4.sh
					fi
				done
			done
		done
	done
done
