#!/bin/sh

export MLP_WDIR=/home/s1045064/dissertation

source /home/s1045064/dissertation/venv/bin/activate

export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MLP_WDIR/repos-3rd/OpenBLAS/lib:/opt/cuDNN-7.0/lib64:

THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32 python /home/s1045064/dissertation/repo-diss/sentence-classification/conv_net_sentence.py -static -word2vec

#THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python /home/s1045064/dissertation/repo-diss/sentence-classification/conv_net_cv_sent.py -static -word2vec


