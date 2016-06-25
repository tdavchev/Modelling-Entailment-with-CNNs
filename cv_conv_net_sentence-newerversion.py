"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf
Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle as pickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import re
import warnings
import sys
import time
import random
warnings.filterwarnings("ignore")

# srng = np.random.RandomState(3435)
srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

# def softmax(X):
#     e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
#     return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    """
    RMSprop update rule, mostly from 
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 =  T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    # pyx = softmax(T.dot(l4, w_o))
    pyx = T.nnet.softmax(T.dot(l4,w_o))
    return l1, l2, l3, l4, pyx

# def shuffle_data(datasets, batch_size, img_h):
#     #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
#     #extra data (at random)
#     np.random.seed(3435)
#     if datasets[0].shape[0] % batch_size > 0:
#         extra_data_num = batch_size - datasets[0].shape[0] % batch_size
#         train_set = np.random.permutation(datasets[0])   
#         extra_data = train_set[:extra_data_num]
#         new_data=np.append(datasets[0],extra_data,axis=0)
#     else:
#         new_data = datasets[0]
#     new_data = np.random.permutation(new_data)
#     n_batches = new_data.shape[0]/batch_size
#     n_train_batches = int(np.round(n_batches*0.9))
#     #divide train set into train/val sets 
#     test_set_x = datasets[1][:,:img_h] 
#     test_set_y = np.asarray(datasets[1][:,-1],"int32")
#     train_set = new_data[:n_train_batches*batch_size,:]
#     val_set = new_data[n_train_batches*batch_size:,:]     
#     train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
#     val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
#     n_val_batches = n_batches - n_train_batches

#     return [[train_set_x, train_set_y], [val_set_x, val_set_y], [test_set_x, test_set_y]]


def build_data_cv(data, labels, cv=10):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    for idx in xrange(0, len(data)):
        # words = set(line[:-1].split())
        datum  = {"y":labels[idx],
                  "text": data[idx][:],
                  "num_words": data[idx][:].shape[0],
                  "split": np.random.randint(0,cv)}
        revs.append(datum)

    return revs

def zero_pad(sent):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    outcome = []
    for s in xrange(0,len(sent)):
        senta = sent[s]
        for  i in xrange(0, 92): # (784-600)/2
            senta = np.append(0,senta)
            senta = np.append(senta,0)
        outcome.append(senta)

    outcome = np.array(outcome,dtype="float")

    return outcome


def make_idx_data_cv(revs, cv):
    """
    Transforms sentences into a 2-d matrix.
    """
    trainX, trainY, testX, testY = [], [], [], []
    for rev in revs:
        if rev["split"]==cv:
            testX.append(rev["text"])
            testY.append([rev["y"]])
        else:
            trainX.append(rev["text"])
            trainY.append(rev["y"])
    trainX = np.array(trainX,dtype="float")
    trainY = np.array(trainY,dtype="int")
    testX = np.array(testX,dtype="float")
    textY = np.array(testY,dtype="int")

    return [trainX, trainY, testX, testY]

def one_hot(data):
    one_hots = []
    for idx in xrange(0,len(data)):
        if data[idx] == 0:
            one_hots.append([1,0,0])
        elif data[idx] == 1:
            one_hots.append([0,1,0])
        else:
            one_hots.append([0,0,1])

    one_hots = np.array(one_hots,dtype="int")

    return one_hots


if __name__=="__main__":
    batch_size = 100
    print "loading data...",
    # obtain the two sentences

    first, second = [], []
    labels = []
    lines = open("first_conv-layer-output.txt").read().splitlines()
    for idx in xrange(0,len(lines)):
        first.append(lines[idx])
        first[idx] = first[idx].strip()
        first[idx] = first[idx].split("  ")
        labels.append(first[idx][-1].split()[1])
        first[idx][-1] = first[idx][-1].split()[0]
        first[idx] = [float(first[idx][i]) for i in xrange(0,len(first[idx]))]

    lines = open("second_conv-layer-output.txt").read().splitlines()
    for idx in xrange(0,len(lines)):
        second.append(lines[idx])
        second[idx] = second[idx].strip()
        second[idx] = second[idx].split("  ")
        second[idx][-1] = second[idx][-1].split()[0]
        second[idx] = [float(second[idx][i]) for i in xrange(0,len(second[idx]))]

    print "data loaded!"
    print "Concatenating..."
    input_data = np.concatenate((first,second),axis=1)
    print "Done."

    revs = build_data_cv(input_data,labels)
    
    results = []
    r = range(0,1) # used in cross validation

    for i in r:

        print "--------------------------"
        print "          CV: {0}".format(i)
        print "--------------------------"

        trX, trY, teX, teY = make_idx_data_cv(revs, i)

        trX = zero_pad(trX)
        teX = zero_pad(teX)

        # trX = trX.reshape(-1,1,24,25)
        # teX = teX.reshape(-1,1,24,25)
        trX = trX.reshape(-1,1,28,28)
        teX = teX.reshape(-1,1,28,28)

        trY = one_hot(trY)
        teY = one_hot(teY)

        X = T.ftensor4()
        Y = T.fmatrix()

        w = init_weights((32, 1, 4, 4))
        w2 = init_weights((64, 32, 4, 4))
        w3 = init_weights((128, 64, 4, 4))
        w4 = init_weights((128 * 2 * 2, 512))
        w_o = init_weights((512, 3))

        noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
        l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
        y_x = T.argmax(py_x, axis=1)


        cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
        params = [w, w2, w3, w4, w_o]
        updates = RMSprop(cost, params, lr=0.001)

        train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)


        if trX.shape[0] % batch_size > 0:
            extra_data_num = batch_size - trX.shape[0] % batch_size
            new_data = []
            new_labels = []
            for k in xrange(0, extra_data_num):
                idx = random.randint(0,trX.shape[0])
                new_data.append(trX[idx])
                new_labels.append(trY[idx])
            trX=np.append(trX,new_data,axis=0)
            trY=np.append(trY,new_labels,axis=0)

        alph = 'ABCDEFGHIJKLMNOPQRSTYVWXYZ'
        for epoch in range(25):
            num_mini_batch = np.ceil(len(trX)/batch_size)
            print epoch,
            for start, end in zip(list(range(0,len(trX), batch_size)), list(range(batch_size, len(trX),batch_size))):
                cost = train(trX[start:end], trY[start:end])
                if (start/batch_size) % np.ceil(num_mini_batch/20) == 0:
                    print(alph[epoch%len(alph)]),
                    sys.stdout.flush()
            print(np.mean(np.argmax(teY, axis=1) == predict(teX)))

    with open('snli.weights','wb') as f:
        pickle.dump(params, f)