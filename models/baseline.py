"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf
Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import theano.tensor.signal.conv
import re
import warnings
import sys
import time
warnings.filterwarnings("ignore")

def train_conv_net(datasets,
                   U,
                   img_w=300,
                   filter_hs=[3,4,5],
                   hidden_units=[100,3],
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25,
                   batch_size=50,
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True,
                   cv=False):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec and GloVe)
    filter_hs = filter window sizes
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """
    print cv
    rng = np.random.RandomState(3435)
    img_h = (len(datasets[0][0])-1)
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch),("activations",activations),("baseline", True)]
    print parameters
    sys.stdout.flush()
    #define model architecture
    print("define model architecture")
    sys.stdout.flush()
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    

    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,(x.shape[1]),Words.shape[1]))
 
    conv_layers = []
    layer1_inputs = []
    # CNN
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)

    ffwd_layer_input = concatenate_tensors(layer1_inputs)
    hidden_units[0] = feature_maps*len(filter_hs) # 300

    classifier = MLPDropout(rng, input=ffwd_layer_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    #define parameters of the model and update functions using adadelta
    print "define parameters of the model and update functions using adadelta"
    sys.stdout.flush()

    params = update_params(classifier, [conv_layers])
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]

    cost = classifier.negative_log_likelihood(y)
    # p_y_given_x = classifier.p_y_given_x

    # weights = classifier.getW()
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
    #extra data (at random)
    np.random.seed(3435)
    new_data = complete_train_data(datasets[0], batch_size)
    # new_data = np.random.permutation(new_data) # makes it all nonsence ! why ?

    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    n_val_batches = n_batches - n_train_batches
    # print n_val_batches

    val_set = new_data[n_train_batches*batch_size:,:]
    # n_train_batches, _ = get_n_batches(new_data, [], batch_size, True)
    # val_set = []
    # if cv:
    #     val_set = new_data[n_train_batches*batch_size:,:]
    # else:
    #     val_set = datasets[2]

    # _,n_val_batches = get_n_batches([], val_set, batch_size, True)

    #divide train set into train/val sets
    print "divide train set into train/val sets"
    sys.stdout.flush()

    test_set_x, test_set_y = process_test(datasets[1])
    train_set_x, train_set_y = process_train(new_data, 
        img_h,
        True, 
        n_train_batches,
        batch_size=batch_size)

    val_set_x, val_set_y = process_valid(val_set[:,:img_h],val_set[:,-1])

    #compile theano functions to get train/val/test errors
    print "compile theano functions to get train/val/test errors"
    sys.stdout.flush()
    val_model = build_model(index, classifier, batch_size, val_set_x, val_set_y, x, y)
    test_model = build_model(index, classifier, batch_size, train_set_x, train_set_y, x, y)
    train_model = build_train_model(index, batch_size, cost, grad_updates, train_set_x, train_set_y, x, y)

    test_ffwd_layer_input = build_test(img_h, test_set_x.shape[0], Words, conv_layers, x)

    test_y_pred = classifier.predict(test_ffwd_layer_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x,y], test_error, allow_input_downcast = True)

    #start training over mini-batches
    print '... training'
    sys.stdout.flush()
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0
    cost_epoch = 0
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index) #2-4 conv 1 is output
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)
        print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.))
        sys.stdout.flush()
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss = test_model_all(test_set_x,test_set_y)
            test_perf = 1 - test_loss

    return test_perf