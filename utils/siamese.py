def concatenate(layers):
    concat= [[],[]]
    for br in xrange(0,2):
        lista = []
        for idx in xrange(0,3):
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
        test_concat = concatenate(test_pred_layers)

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