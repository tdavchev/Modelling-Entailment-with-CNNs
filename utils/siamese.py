def concatenate(layers):
    concat= [[],[]]
    for br in xrange(0,2):
        lista = []
        for idx in xrange(0,len(layers[br])):
            lista.append(layers[br][idx])

        concat[br] = T.concatenate(lista,1)

    return concat

def set_test_params(mode, test_pred_layers_one=[],test_pred_layers_two=[]):
    test_concat = [[],[]]
    if mode == "concat":
        img_w = 50
        img_h = 12

    elif mode == "mix1" or mode == "mix2":
        img_w = 80
        img_h = 15

    else:
        img_w = 30
        img_h = 10

    return [test_concat, img_w, img_h]

def populate_pred_layers(mode,conv_layers,test_layer0_input,test_size):
    if mode == "concat":
        test_pred_layers = []
    else:
        test_pred_layers = [[], []] 

    for idx in xrange(0,2):
        for conv_layer in conv_layers[idx]:
            test_layer0_output = conv_layer.predict(test_layer0_input[idx], test_size)
            if mode == "concat":
                test_pred_layers.append(test_layer0_output.flatten(2))
            else:
                test_pred_layers[idx].append(test_layer0_output.flatten(2)) 

    return test_pred_layers

def set_layer1_input(mode,test_pred_layers,test_concat, img_h, img_w, data, alpha, beta):
    if mode == "concat":
        test_layer1_input = concatenate_tensors(test_pred_layers)
    else:
        test_concat = concatenate(test_pred_layers) # !!!!!!!!!!!!!! only the first two CNNs

        if mode == "mul":
            test_layer1_input = mul(test_concat)

        elif mode == "add":
            test_layer1_input = add(len(data[:]), alpha, beta, test_concat)

        elif mode == "sub":
            test_layer1_input = sub(len(data[:]), alpha, beta, test_concat)

        elif mode == "circ":
            test_layer1_input=circular_convolution(test_concat)

        elif mode == "mix1":
            test_pred_inputs = []
            for idx in xrange(0,2):
                for br in xrange(0,3):
                    test_pred_inputs.append(test_pred_layers[idx][br])

            test_layer1_input = mix1(test_pred_inputs,len(data[:]),alpha,beta,test_concat)

        elif mode == "mix2":
            test_pred_inputs = []
            for idx in xrange(0,2):
                for br in xrange(0,3):
                    test_pred_inputs.append(test_pred_layers[idx][br])

            test_layer1_input = mix2(test_pred_inputs,len(data[:]),alpha,beta,test_concat)

        elif mode == "mix3":
            test_pred_inputs = []
            for idx in xrange(0,2):
                for br in xrange(0,3):
                    test_pred_inputs.append(test_pred_layers[idx][br])

            test_layer1_input = mix3(test_pred_inputs,len(data[:]),alpha,beta,test_concat)

    return test_layer1_input.reshape((-1,img_h,img_w))

def set_lengths(modeOp):
    # keep relatively close ratio to 300/81
    img_w = 0
    img_h = 0
    if modeOp == "concat":
        img_w = 50
        img_h = 12
    elif modeOp == "mix1" or modeOp == "mix2" or modeOp == "mix3":
        img_w = 80
        img_h = 15
    else:
        img_w = 30
        img_h = 10

    return img_w,img_h

def build_test(img_h,img_w, test_size, Words, conv_layers, x, mode, data, alpha, beta):
    # initialize layer 0's input
    test_layer0_input = set_layer0_input(Words, img_h, test_size, x)
    # initialize new parameters
    test_concat, img_w, img_h = set_test_params(mode)
    # populate layers
    test_pred_layers = populate_pred_layers(mode, conv_layers, test_layer0_input, test_size)
    # initialize layer 1's input
    test_layer1_input = set_layer1_input(mode, test_pred_layers, test_concat, img_h, img_w, data, alpha, beta)
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

    return [train[:2000], test[:1200], valid[:1200]]

def set_layer0_input(Words,img_h,test_size,x):
    test_layer0_input_one = Words[T.cast(x[:,:89].flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    test_layer0_input_two = Words[T.cast(x[:,89:].flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    
    return [test_layer0_input_one,test_layer0_input_two]