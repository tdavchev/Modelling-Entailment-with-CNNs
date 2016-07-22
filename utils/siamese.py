def concatenate(layers):
    concat= [[],[]]
    for br in xrange(0,2):
        lista = []
        for idx in xrange(0,len(layers[br])):
            lista.append(layers[br][idx])

        concat[br] = T.concatenate(lista,1)

    return concat

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

def set_layer1_input(mode,test_pred_layers,test_concat, img_h, img_w, data_len, alpha, beta, num_maps):
    if mode == "concat":
        test_layer1_input = concatenate_tensors(test_pred_layers)
    else:
        test_concat = concatenate(test_pred_layers)

        if mode == "mul":
            test_layer1_input = mul(test_concat)

        elif mode == "add":
            test_layer1_input = add(data_len, alpha, beta, test_concat)

        elif mode == "sub":
            test_layer1_input = sub(data_len, alpha, beta, test_concat)

        elif mode == "circ":
            test_layer1_input=circular_convolution(test_concat)

        elif mode == "mix1":
            test_pred_inputs = []
            for idx in xrange(0,2):
                for br in xrange(0,num_maps):
                    test_pred_inputs.append(test_pred_layers[idx][br])
                    
            test_layer1_input = mix1(test_pred_inputs,data_len,alpha,beta,test_concat)

        elif mode == "mix2":
            test_pred_inputs = []
            for idx in xrange(0,2):
                for br in xrange(0,num_maps):
                    test_pred_inputs.append(test_pred_layers[idx][br])

            test_layer1_input = mix2(test_pred_inputs,data_len,alpha,beta,test_concat)

        elif mode == "mix3":
            test_pred_inputs = []
            for idx in xrange(0,2):
                for br in xrange(0,num_maps):
                    test_pred_inputs.append(test_pred_layers[idx][br])

            test_layer1_input = mix3(test_pred_inputs,data_len,alpha,beta,test_concat)

        elif mode == "mix4":
            test_pred_inputs = []
            for idx in xrange(0,2):
                for br in xrange(0,num_maps):
                    test_pred_inputs.append(test_pred_layers[idx][br])

            test_layer1_input = mix4(test_pred_inputs,data_len,alpha,beta,test_concat)

        elif mode == "mix5":
            test_pred_inputs = []
            for idx in xrange(0,2):
                for br in xrange(0,num_maps):
                    test_pred_inputs.append(test_pred_layers[idx][br])

            test_layer1_input = mix5(test_pred_inputs,data_len,alpha,beta,test_concat)

        elif mode == "mix6":
            test_pred_inputs = []
            for idx in xrange(0,2):
                for br in xrange(0,num_maps):
                    test_pred_inputs.append(test_pred_layers[idx][br])

            test_layer1_input = mix6(test_pred_inputs,data_len,alpha,beta,test_concat)

        elif mode == "mix7":
            test_pred_inputs = []
            for idx in xrange(0,2):
                for br in xrange(0,num_maps):
                    test_pred_inputs.append(test_pred_layers[idx][br])
                    
            test_layer1_input = mix7(test_pred_inputs,data_len,alpha,beta,test_concat)

    return test_layer1_input.reshape((-1,img_h,img_w))

def set_lengths(modeOp, num_maps):
    # keep relatively close ratio to 300/81
    img_w = 0
    img_h = 0
    if modeOp == "concat" or modeOp == "mix4" or modeOp == "mix5" or modeOp == "mix6":
        img_h = 50#25#50#100
        if num_maps == 3:
            img_w = 12#24#12#6
        elif num_maps == 4:
            img_w = 16#32#16#8

    elif modeOp == "mix1" or modeOp == "mix2" or modeOp == "mix3":
        img_h = 25#50#25#50#100#200
        if num_maps == 3:
            img_w = 56#28#14#7
        elif num_maps == 4:
            img_w = 64#32#16#8

    elif modeOp == "mix7":
        img_h = 25
        if num_maps == 3:
            img_w = 60
        elif num_maps == 4:
            img_w = 80 

    else:
        img_w = 10
        if num_maps == 3:
            img_h = 30
        elif num_maps == 4:
            img_h = 40

    print img_w,img_h
    return img_w,img_h

def build_test(img_h,img_w, test_size, Words, conv_layers, x, mode, data_len, alpha, beta, num_maps):
    # initialize layer 0's input
    test_layer0_input = set_layer0_input(Words, img_h, test_size, x)
    # initialize new parameters
    img_w, img_h = set_lengths(mode, num_maps)
    # populate layers
    test_pred_layers = populate_pred_layers(mode, conv_layers, test_layer0_input, test_size)
    # initialize layer 1's input
    test_layer1_input = set_layer1_input(mode, test_pred_layers, [[],[]], img_h, img_w, data_len, alpha, beta, num_maps)
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

    return [train, test, valid]

def set_layer0_input(Words,img_h,test_size,x):
    test_layer0_input_one = Words[T.cast(x[:,:x.shape[1]/2].flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    test_layer0_input_two = Words[T.cast(x[:,x.shape[1]/2:].flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    
    return [test_layer0_input_one,test_layer0_input_two]
