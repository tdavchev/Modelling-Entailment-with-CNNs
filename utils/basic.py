'''
Basic model specific helper functions
'''

def build_test(img_h, test_size, Words, conv_layers, x):
    """
    Method that builds the test after training and validation stage

    params: imgage height, test size, Words, convolution layers, x
    return: feed-forward layer input
    """
    test_layer0_input = set_layer0_input(Words,img_h,test_size,x) # initialize layer 0's input
    test_pred_layers = [] # predict with CNN
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))

    ffwd_layer_input = concatenate_tensors(test_pred_layers)

    return ffwd_layer_input

def make_idx_data(revs, word_idx_map, max_l=81, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, valid, test = [], [], []
    for idx in xrange(0,len(revs)):
        sent = get_idx_from_sent(revs[idx]["text"], word_idx_map, max_l, k, filter_h)
        sent.append(int(revs[idx]["label"]))
        if idx > 0:
            if revs[idx]["type"]=="test":
                test.append(sent)
            elif revs[idx]["type"]=="train":
                train.append(sent)
            else:
                valid.append(sent)

    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    valid = np.array(valid,dtype="int")

    return [train, test, valid]

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
    return [train, test] 

def set_layer0_input(Words, img_h, test_size, x):
    test_layer0_input_one = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
   
    return test_layer0_input_one
