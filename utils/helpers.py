def process_train(new_data, img_h=0, cv=False, n_train_batches=0, batch_size=0):
    if cv:
        train_set = new_data[:n_train_batches*batch_size,:]
        train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
    else:
        train_set = new_data[:,:]
        train_set_x, train_set_y = shared_dataset((train_set[:,:-1],train_set[:,-1]))

    return train_set_x, train_set_y


def process_test(test_data):
    test_set_x = test_data[:,:-1]
    test_set_y = np.asarray(test_data[:,-1],"int32")

    return test_set_x, test_set_y

def process_valid(data,labels):
    val_set_x, val_set_y = shared_dataset((data,labels))

    return val_set_x, val_set_y

def get_n_batches(new_data, valid_data, batch_size, cv=False):
    if cv:
        new_data = np.random.permutation(new_data)
        n_batches = new_data.shape[0]/batch_size
        n_train_batches = int(np.round(n_batches*0.9))
        n_val_batches = n_batches - n_train_batches

    else:
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

def update_params(classifier, conv_layers):
    params = classifier.params
    for num_conv_layer in conv_layers:
        for conv_layer in num_conv_layer:
            params += conv_layer.params

    return params

def set_layer0_input(Words,img_h,test_size,x, cv=False):
    if cv:
        return Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    else:
        test_layer0_input_one = Words[T.cast(x[:,:89].flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
        test_layer0_input_two = Words[T.cast(x[:,89:].flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    
    return [test_layer0_input_one,test_layer0_input_two]

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

    return [train[:100], test[:10], valid[:10]]

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        if rev["split"]==cv:            
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train[:10], test[:10]] 


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

def build_test(img_h,img_w, test_size, Words, conv_layers,x, mode, data, alpha, beta):
    # initialize layer 0's input
    test_layer0_input = set_layer0_input(Words,img_h,test_size,x)
    # initialize new parameters
    test_concat, img_w, img_h = set_test_params(mode)
    # populate layers
    test_pred_layers = populate_pred_layers(mode,conv_layers,test_layer0_input,test_size)
    # initialize layer 1's input
    test_layer1_input = set_layer1_input(mode,test_pred_layers,test_concat, img_h, img_w, data,alpha,beta)
    # reshape for third CNN
    test_layer0_input_three = test_layer1_input.reshape(
        (test_layer1_input.shape[0],
            1,
            test_layer1_input.shape[1],
            test_layer1_input.shape[2]
            )
        )
    # predict with third CNN
    test_pred_layers = []
    for conv_layer in conv_layers[-1]:
        test_layer0_output = conv_layer.predict(test_layer0_input_three, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))

    ffwd_layer_input = T.concatenate(test_pred_layers,1)

    return ffwd_layer_input