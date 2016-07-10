#!/bin/bash
export PATH=$PATH:/opt/cuda-7.5.18/bin:/opt/cuDNN-7.0:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s1579267/tools/OpenBLAS/lib:/opt/cuda-7.5.18/lib64:/opt/cuDNN-7.0/lib64:/opt/cuDNN-7.0:
export CUDA_ROOT=/opt/cuda-7.5.18
export MLP_WDIR=/home/s1045064/dissertation

cd /home/s1045064/dissertation/repo-diss/sentence-classification/
source /home/s1045064/dissertation/venv/bin/activate

echo "Sentence Classification Benchmark"

THEANO_FLAGS="device=gpu$GPU_NO,mode=FAST_RUN,floatX=float32" python /home/s1045064/dissertation/repo-diss/sentence-classification/baseline.py -static -word2vec $BATCH_SIZE_F $DROPOUT_F $CNL_F $LR_DECAY $ACTIVATION $SQR_NORM_LIM
