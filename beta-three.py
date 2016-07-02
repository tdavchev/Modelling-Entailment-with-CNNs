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
import re
import warnings
import sys
import time
import cv_process_data as process
warnings.filterwarnings("ignore")

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

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
                   modeOp="mul",
                   alpha=1,
                   beta=1,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec and GloVe)
    filter_hs = filter window sizes
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """
    rng = np.random.RandomState(3435)
    img_h = (len(datasets[0][0])-1)/2
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
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch),("mode",modeOp), ("alpha",alpha),("beta",beta),("activations",activations)]
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
    

    first_layer0_input = Words[T.cast(x[:,:89].flatten(),dtype="int32")].reshape((x.shape[0],1,(x.shape[1]/2),Words.shape[1]))
    second_layer0_input = Words[T.cast(x[:,89:].flatten(),dtype="int32")].reshape((x.shape[0],1,(x.shape[1]/2),Words.shape[1]))
    
    first_conv_layers = []
    second_conv_layers = []
    layer1_inputs = []
    one_layers = []
    two_layers = []

    # FIRST CNN
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=first_layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        first_conv_layers.append(conv_layer)
        one_layers.append(layer1_input)
        layer1_inputs.append(layer1_input)


    # SECOND CNN
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=second_layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        second_conv_layers.append(conv_layer)
        two_layers.append(layer1_input)
        layer1_inputs.append(layer1_input)


    if modeOp == "concat":
        layer1_input = T.concatenate(layer1_inputs,1)
        # keep relatively close ratio to 300/81
        img_w = 50
        img_h = 12
    else:
        lista =[]
        layer1_input = []
        layer1_inputtttttt = []
        for idx in xrange(0,3):
            lista.append(one_layers[idx])

        one_layers = T.concatenate(lista,1)
        lista = []

        img_w = 30
        img_h = 10
        for idx in xrange(0,3):
            lista.append(two_layers[idx])

        two_layers = T.concatenate(lista,1)
        lista = []
        
        if modeOp == "add":
            a = np.ndarray(shape=(batch_size,img_w), dtype='float32')
            b = np.ndarray(shape=(batch_size,img_w), dtype='float32')

            a.fill(alpha)
            b.fill(beta)

            one_layers = T.mul(one_layers,a)
            two_layers = T.mul(two_layers,b)

            layer1_input = T.add(one_layers,two_layers) # [50 300]
        elif modeOp == "mul":
            layer1_input = T.mul(one_layers,two_layers)
            
    print img_h,img_w
    layer1_cnn_input = layer1_input.reshape((-1,img_h,img_w))
        
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    
    third_layer0_input = layer1_cnn_input.reshape((layer1_cnn_input.shape[0],1,layer1_cnn_input.shape[1],layer1_cnn_input.shape[2]))

    third_conv_layers = []
    layer1_inputs = []

    # THIRD CNN
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=third_layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        third_conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)

    ffwd_layer_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs) # 300

    classifier = MLPDropout(rng, input=ffwd_layer_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    #define parameters of the model and update functions using adadelta
    print "define parameters of the model and update functions using adadelta"
    sys.stdout.flush()

    params = update_params(classifier, [first_conv_layers, second_conv_layers, third_conv_layers])
    
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]

    cost = classifier.negative_log_likelihood(y)
    p_y_given_x = classifier.p_y_given_x

    # weights = classifier.getW()
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
    #extra data (at random)
    np.random.seed(3435)
    new_data = complete_train_data(datasets[0], batch_size)
    new_data = np.random.permutation(new_data)

    n_train_batches, n_val_batches = get_n_batches(new_data, datasets[1], batch_size)

    #divide train set into train/val sets
    print "divide train set into train/val sets"
    sys.stdout.flush()

    test_set_x, test_set_y = process_test(datasets[2])
    train_set_x, train_set_y = process_train(new_data)
    val_set_x, val_set_y = process_valid(datasets[1])

    #compile theano functions to get train/val/test errors
    print "compile theano functions to get train/val/test errors"
    sys.stdout.flush()
    val_model = build_model(index, classifier, batch_size, val_set_x, val_set_y, x, y)
    test_model = build_model(index, classifier, batch_size, train_set_x, train_set_y, x, y)
    train_model = build_train_model(index, batch_size, cost, grad_updates, train_set_x, train_set_y, x, y)

    img_h = (len(datasets[0][0])-1)/2
    # if modeOp == "concat":
    #     ffwd_layer_input = build_test(img_h, test_set_x.shape[0], Words, [first_conv_layers, second_conv_layers, third_conv_layers],x)
    # else:
    ffwd_layer_input = build_test(img_h, img_w, test_set_x.shape[0], Words, [first_conv_layers, second_conv_layers, third_conv_layers],x, modeOp)

    test_y_pred = classifier.predict(ffwd_layer_input)
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
                # cost_epoch = train_model(minibatch_index) #2-4 conv 1 is output
                cost_epoch = train_model(minibatch_index)
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
            test_perf = 1- test_loss

    return test_perf

def process_train(new_data):
    train_set = new_data[:,:]
    train_set_x, train_set_y = shared_dataset((train_set[:,:-1],train_set[:,-1]))

    return train_set_x, train_set_y

def process_test(test_data):
    test_set_x = test_data[:,:-1]
    test_set_y = np.asarray(test_data[:,-1],"int32")

    return test_set_x, test_set_y

def process_valid(valid_data):
    val_set = valid_data
    val_set_x, val_set_y = shared_dataset((val_set[:,:-1],val_set[:,-1]))

    return val_set_x, val_set_y

def get_n_batches(new_data, valid_data, batch_size):
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches))
    n_val_batches = valid_data.shape[0]/batch_size

    return n_train_batches, n_val_batches


def complete_train_data(data, batch_size):
    print "shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate"
    sys.stdout.flush()
    np.random.seed(3435)
    if data.shape[0] % batch_size > 0:
        extra_data_num = batch_size - data.shape[0] % batch_size
        train_set = np.random.permutation(data) # no need to store ... I use seed so it will be always the same
        extra_data = train_set[:extra_data_num]
        new_data=np.append(data,extra_data,axis=0)
    else:
        new_data = data

    return new_data

def build_model(index, classifier, batch_size, set_x, set_y, x, y):
    model = theano.function([index], classifier.errors(y),
             givens={
                x: set_x[index * batch_size: (index + 1) * batch_size],
                 y: set_y[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)

    return model

def build_train_model(index, batch_size, cost, grad_updates, train_set_x, train_set_y, x, y):
    train_model = theano.function([index], cost, updates=grad_updates,
      givens={
        x: train_set_x[index*batch_size:(index+1)*batch_size],
          y: train_set_y[index*batch_size:(index+1)*batch_size]},
                              allow_input_downcast = True)
    
    return train_model

def update_params(classifier, conv_layers):
    params = classifier.params
    for num_conv_layer in conv_layers:
        for conv_layer in num_conv_layer:
            params += conv_layer.params

    return params

# def concatenate(layers,cases):
#     lista = []
#     for idx in xrange(0,cases):
#         lista.append(layers[idx])

#     pred_layers = T.concatenate(lista,1)
#     return pred_layers



def build_test(img_h,img_w, test_size, Words, conv_layers,x, mode):
    test_pred_layers_one, test_pred_layers_two = [], []
    test_layer0_input_one = Words[T.cast(x[:,:89].flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    test_layer0_input_two = Words[T.cast(x[:,89:].flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))

    test_layer0_input = [test_layer0_input_one,test_layer0_input_two]
    if mode != "concat":
        test_pred_layers = [test_pred_layers_one, test_pred_layers_two]
        test_concat = [[],[]]
        img_w = 30
        img_h = 10
    else:
        test_pred_layers = []
        img_w = 50
        img_h = 12

    for idx in xrange(0,2):
        for conv_layer in conv_layers[idx]:
            test_layer0_output = conv_layer.predict(test_layer0_input[idx], test_size)
            if mode == "concat":
                test_pred_layers.append(test_layer0_output.flatten(2))
            else:
                test_pred_layers[idx].append(test_layer0_output.flatten(2)) 

    if mode == "concat":
        test_layer1_input = T.concatenate(test_pred_layers, 1)
    else:
        for br in xrange(0,2):
            lista = []
            for idx in xrange(0,3):
                lista.append(test_pred_layers[br][idx])

            test_concat[br] = T.concatenate(lista,1)

        if mode == "mul":
            test_layer1_input = T.mul(test_pred_layers[0],test_pred_layers[1])
        elif mode == "add":
            test_a = np.ndarray(shape=(len(datasets[2][:]),300), dtype='float32')
            test_b = np.ndarray(shape=(len(datasets[2][:]),300), dtype='float32')
            
            test_a.fill(alpha)
            test_b.fill(beta)
            
            test_pred_layers_one = T.mul(test_pred_layers_one,test_a)
            test_pred_layers_two = T.mul(test_pred_layers_two,test_b)

            test_layer1_input = T.add(test_pred_layers_one,test_pred_layers_two)
    
    test_layer1_cnn_input = test_layer1_input.reshape((-1,img_h,img_w))


    test_layer0_input_three = test_layer1_cnn_input.reshape((test_layer1_cnn_input.shape[0],1,test_layer1_cnn_input.shape[1],test_layer1_cnn_input.shape[2]))
    test_pred_layers = []

    for conv_layer in conv_layers[-1]:
        test_layer0_output = conv_layer.predict(test_layer0_input_three, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))

    ffwd_layer_input = T.concatenate(test_pred_layers,1)

    return ffwd_layer_input

# def build_test(img_h, test_size, Words, conv_layers,x):
#     test_pred_layers = []
#     # img_h = (len(datasets[0][0])-1)/2
#     # test_size = test_set_x.shape[0]
#     test_layer0_input_one = Words[T.cast(x[:,:89].flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
#     test_layer0_input_two = Words[T.cast(x[:,89:].flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))

#     test_layer0_input = [test_layer0_input_one,test_layer0_input_two]
    
#     for idx in xrange(0,2):
#         for conv_layer in conv_layers[idx]:
#             test_layer0_output = conv_layer.predict(test_layer0_input[idx], test_size)
#             test_pred_layers.append(test_layer0_output.flatten(2))

#     test_layer1_input = T.concatenate(test_pred_layers, 1)
#     test_layer1_cnn_input = test_layer1_input.reshape((-1,12,50))


#     img_w = 50
#     img_h = 12

#     test_layer0_input_three = test_layer1_cnn_input.reshape((test_layer1_cnn_input.shape[0],1,test_layer1_cnn_input.shape[1],test_layer1_cnn_input.shape[2]))
#     test_pred_layers = []

#     for conv_layer in conv_layers[-1]:
#         test_layer0_output = conv_layer.predict(test_layer0_input_three, test_size)
#         test_pred_layers.append(test_layer0_output.flatten(2))

#     ffwd_layer_input = T.concatenate(test_pred_layers,1)

#     return ffwd_layer_input


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to

def get_idx_from_sent(sent, word_idx_map, max_l=81, k=300, filter_h=5, padit = True):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    if padit:
        x = []
        pad = filter_h - 1
        for i in xrange(pad):
            x.append(0)

    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

# def make_idx_data(revs, word_idx_map, cur_idx, max_l=81, k=300, filter_h=5):
#     """
#     Transforms sentences into a 2-d matrix.
#     """
#     train, valid, test = [], [], []
#     for rev in revs:
#         if(rev["idx"]==cur_idx):
#             sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
#             sent.append(rev["label"])
#             if rev["type"]=="test":
#                 test.append(sent)
#             elif rev["type"]=="train":
#                 train.append(sent)
#             else:
#                 valid.append(sent)
#     train = np.array(train,dtype="int")
#     test = np.array(test,dtype="int")
#     valid = np.array(valid,dtype="int")

#     return [train, valid, test]

def make_idx_data(revs, word_idx_map, max_l=81, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, valid, test = [], [], []
    sent = []
    count = 0
    for idx in xrange(0,len(revs)):
        if ((idx % 2)==0):
            sent = get_idx_from_sent(revs[idx]["text"], word_idx_map, max_l, k, filter_h)
        else:
            sentApp = get_idx_from_sent(revs[idx]["text"], word_idx_map, max_l, k, filter_h)
            sentApp.append(int(revs[idx]["label"]))
            sent = np.concatenate((sent,sentApp),axis=0) # 89*2
            if idx > 0:
                if revs[idx]["type"]=="test":
                    test.append(sent)
                elif revs[idx]["type"]=="train":
                    count += 1
                    train.append(sent)
                else:
                    valid.append(sent)

                sent = []

    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    valid = np.array(valid,dtype="int")

    return [train[:1000], valid[:120], test[:120]]

def store_sent(batches, num, datasets):
    p_sento_finale = []
    sento_finale = []
    print "sentence {0} batch len {1}".format(num, len(batches))
    for batch in xrange(0, len(batches)):
        for sentence_id in xrange(0,len(batches[batch])):
            if len(datasets[0]) > sentence_id:
                off = []
                p_off = []
                off = np.append(batches[batch][sentence_id],datasets[0][sentence_id,-1])
                p_off = np.append(batches[batch][sentence_id],(datasets[0][sentence_id,-1]))
                sento_finale.append(off)
                p_sento_finale.append(p_off)

    print "sentences in {0} concatenated. {1}".format(num, len(first_sent[0]))
    sys.stdout.flush()

    return sento_finale, p_sento_finale

def file_save(sentences, file_name):
    file = file_name+".txt"
    f = open(file,"w")
    print "Saving into text files"
    sys.stdout.flush()
    for sent in xrange(0, len(sentences)):
        for br in xrange(0,len(sentences[sent])):
            if (br+1)==len(sentences[sent]):
                f.write('%d' % sentences[sent][br])
            else:
                f.write('%10.6f ' % sentences[sent][br])

        f.write("\n")
    f.close()

def store_output(first_sent, second_sent, datasets):
    s1, p1 = store_sent(first_sent, 0, datasets)
    s2, p2 = store_sent(second_sent, 1, datasets)
    file_save(s1, "first_conv-layer-output")
    file_save(p1, "first_conv-layer-output-prob")
    file_save(s1, "second_conv-layer-output")
    file_save(p1, "second_conv-layer-output-prob")

if __name__=="__main__":
    print "loading data..."
    sys.stdout.flush()
    x = cPickle.load(open("snli-GloVe-Split.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    sys.stdout.flush()
    mode= sys.argv[1]
    word_vectors = sys.argv[2]
    # mode = "-nonstatic"
    # word_vectors = "-word2vec"
    # batch_size_f = 50
    # dropout_rate_f = 0.5
    # conv_non_linear_f = "relu"

    # Parameters

    batch_size_f = sys.argv[3]
    batch_size_f = int(batch_size_f)
    dropout_rate_f = sys.argv[4]
    dropout_rate_f = float(dropout_rate_f)
    dropout_rate_f /= 100 
    conv_non_linear_f = sys.argv[5]
    modeOp = sys.argv[6]
    lr_decay = sys.argv[7]
    lr_decay = float(lr_decay)
    lr_decay /= 100
    alpha = sys.argv[8]
    alpha = float(alpha)
    alpha /= 100
    beta = sys.argv[9]
    beta = float(beta)
    beta /= 100
    whichAct = sys.argv[10]
    whichAct = int(whichAct)-1
    sqr_norm_lim = sys.argv[11]
    sqr_norm_lim = int(sqr_norm_lim)
    

    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        sys.stdout.flush()
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        sys.stdout.flush()
        non_static=False
    first_sent = [] # note currently takes the output of the convolution layers and not the predictions
    second_sent = [] # I need to try it with predictions as well
    execfile("conv_net_classes.py")
    if word_vectors=="-rand":
        print "using: random vectors"
        sys.stdout.flush()
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        sys.stdout.flush()
        U = W
    results = []
    datasets = make_idx_data(revs, word_idx_map, max_l=81,k=300, filter_h=5)
    print "datasets configured."
    sys.stdout.flush()
    print conv_non_linear_f
    sys.stdout.flush()
    print non_static, batch_size_f,dropout_rate_f, len(datasets[0])
    sys.stdout.flush()
    activations = [ReLU, Sigmoid, Tanh, Iden]
    perf= train_conv_net(datasets,
       U,
       img_w=300,
       filter_hs=[3,4,5],
       hidden_units=[100,3],
       dropout_rate=[dropout_rate_f],
       shuffle_batch=True,
       n_epochs=25,
       batch_size=batch_size_f,
       lr_decay = lr_decay,
       conv_non_linear=conv_non_linear_f,
       activations=[activations[whichAct]],
       sqr_norm_lim=sqr_norm_lim,
       non_static=non_static,
       modeOp=modeOp,
       alpha=alpha,
       beta=beta)


    results.append(perf)
    print str(np.mean(results))
    sys.stdout.flush()
    
    # store_output(first_sent, second_sent, datasets)
    #print "concatenating the two sentences {0}".format(len(first_sent[0]))
    # sys.stdout.flush()
