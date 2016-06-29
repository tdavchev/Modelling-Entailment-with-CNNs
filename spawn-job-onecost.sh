#!/bin/bash:
GPU=0
cnl="relu"
lr_decay=0.95
mode="add"
for bs in 50; do  #150 250; do
   for dropout in 0.2 ; do # 0.5 0.1 0.0
		GPU=$((GPU+1))
		if [ $GPU -eq 3 ]; then
			GPU=0
		fi
      	if [ $1 -eq 1 ]; then
      		echo "Starting Model 2 with 3 CNNs"
			qsub -v BATCH_SIZE_F=$1,DROPOUT_F=$2,CNL_F=$3,GPU_NO=$4 job3.sh
		else
			# if [ $lr_decay != 0.9 ] && [ $cnl != "relu" ] && [ $mode != "mul" ]; then
			# 	echo "tuk sum"
			# 	echo "$cnl $lr_decay $mode"
			# 		echo "sega tuk"
			# 		echo "$cnl $lr_decay $mode"
			# 		if [ $lr_decay != 0.95 ] && [ $cnl != "relu" ] && [ $mode != "mul" ]; then
			# 			echo "i sega tuk"
			# 			echo "$cnl $lr_decay $mode"
			# 			if [ $lr_decay != 0.9 ] && [ $cnl != "tanh" ] && [ $mode != "mul" ]; then
			cnl="relu"
			lr_decay=0.95
			mode="add"
			echo "Starting job with batch size: $bs, dropout: $dropout, conv_non_linear: $cnl lr_decay: $lr_decay in mode: $mode on GPU: $GPU"
			echo "Starting Model 3 with MODE $mode"
			# FIRST='qsub -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh'
			# echo $FIRST
			qsub -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh
			cnl="tanh"
			lr_decay=0.95
			mode="add"
			echo "Starting job with batch size: $bs, dropout: $dropout, conv_non_linear: $cnl lr_decay: $lr_decay in mode: $mode on GPU: $GPU"
			echo "Starting Model 3 with MODE $mode"
			# SECOND='qsub -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh'
			# echo $SECOND
			qsub -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh
			cnl="tanh"
			lr_decay=0.95
			mode="mul"
			echo "Starting job with batch size: $bs, dropout: $dropout, conv_non_linear: $cnl lr_decay: $lr_decay in mode: $mode on GPU: $GPU"
			echo "Starting Model 3 with MODE $mode"
			# THIRD='qsub -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh'
			qsub -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh
			# echo $THIRD
			cnl="tanh"
			lr_decay=0.94
			mode="mul"
			echo "Starting job with batch size: $bs, dropout: $dropout, conv_non_linear: $cnl lr_decay: $lr_decay in mode: $mode on GPU: $GPU"
			echo "Starting Model 3 with MODE $mode"
			qsub -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh
			# FOURTH='qsub -W depend=afterok:$FIRST -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh'
			# echo $FOURTH
			cnl="tanh"
			lr_decay=0.94
			mode="add"
			echo "Starting job with batch size: $bs, dropout: $dropout, conv_non_linear: $cnl lr_decay: $lr_decay in mode: $mode on GPU: $GPU"
			echo "Starting Model 3 with MODE $mode"
			# FIFTH='qsub -W depend=afterok:$SECOND -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh'
			qsub -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh
			# echo $FIFTH
			cnl="relu"
			lr_decay=0.94
			mode="add"
			echo "Starting job with batch size: $bs, dropout: $dropout, conv_non_linear: $cnl lr_decay: $lr_decay in mode: $mode on GPU: $GPU"
			echo "Starting Model 3 with MODE $mode"
			# SIXTH='qsub -W depend=afterok:$THIRD -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh'
			qsub -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh
			# echo $SIXTH
		    cnl="relu"
			lr_decay=0.94
			mode="mul"
			echo "Starting job with batch size: $bs, dropout: $dropout, conv_non_linear: $cnl lr_decay: $lr_decay in mode: $mode on GPU: $GPU"
			echo "Starting Model 3 with MODE $mode"
			qsub -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh
			# SEVENTH='qsub -v BATCH_SIZE_F=$bs,DROPOUT_F=$dropout,CNL_F=$cnl,GPU_NO=$GPU,MODE=$mode,LR_DECAY=$lr_decay job4.sh'
			# echo $SEVENTH
			# 			fi
			# 		fi
			# 	fi
			# fi					
		fi
	done
done
