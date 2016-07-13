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

def concatenate_tensors(layer1_inputs):
    return T.concatenate(layer1_inputs,1)

def add(batch_size, alpha, beta, concat):
    a = np.ndarray(shape=(batch_size,300), dtype='float32')
    b = np.ndarray(shape=(batch_size,300), dtype='float32')

    a.fill(alpha)
    b.fill(beta)

    one_layers = T.mul(concat[0],a)
    two_layers = T.mul(concat[1],b)

    return T.add(one_layers,two_layers) # [50 300]

def sub(batch_size, alpha, beta, concat):
    a = np.ndarray(shape=(batch_size,300), dtype='float32')
    b = np.ndarray(shape=(batch_size,300), dtype='float32')

    a.fill(alpha)
    b.fill(beta)

    one_layers = T.mul(concat[0],a)
    two_layers = T.mul(concat[1],b)

    return T.sub(one_layers,two_layers) # [50 300]

def mul(concat):
    return T.mul(concat[0],concat[1]) # [50,300]

def circular_convolution(concat):
    bs,w=concat[0].shape # [batch,width] [50, 300]

    corr_expr = T.signal.conv.conv2d(
        concat[0], 
        concat[1][::-1].reshape((1, -1)), # reverse the second vector
        image_shape=(1, w), 
        border_mode='full')

    corr_len = corr_expr.shape[1]

    pad = w - corr_len%w    
    v_padded = T.concatenate([corr_expr, T.zeros((bs, pad))], axis=1)

    circ_corr_exp = T.sum(v_padded.reshape((bs, v_padded.shape[1] // w, w)), axis=1)

    return circ_corr_exp[:, ::-1] # [50,300]

def mix1(layer1_inputs,batch_size,alpha,beta,concat):
    layer1_concat = concatenate_tensors(layer1_inputs) # [50 600]
    layer1_sub = sub(batch_size, alpha, beta, concat) # [50 300]

    lista = []
    lista.append(layer1_concat)
    lista.append(layer1_sub)
    layer1_input = T.concatenate(lista,1) # [50 900]

    layer1_mul = mul(concat) # [50 300]

    lista = []
    lista.append(layer1_input)
    lista.append(layer1_mul)

    return T.concatenate(lista,1) # [50 1200]  

def mix2(layer1_inputs,batch_size,alpha,beta,concat):
    layer1_concat = concatenate_tensors(layer1_inputs) # [50 600]
    layer1_sub = sub(batch_size, 1, 1, concat) # [50 300]

    lista = []
    lista.append(layer1_concat)
    lista.append(layer1_sub)
    layer1_input = T.concatenate(lista,1) # [50 900]

    layer1_mul = mul(concat) # [50 300]
    #layer1_add = add(batch_size, alpha, beta, concat) # [50 300]

    lista = []
    lista.append(layer1_input)
    lista.append(layer1_mul)

    return T.concatenate(lista,1) # [50 1200]  

def mix3(layer1_inputs,batch_size,alpha,beta,concat):
    layer1_concat = concatenate_tensors(layer1_inputs) # [50 600]
    layer1_sub = sub(batch_size, 1, 1, concat) # [50 300]

    lista = []
    lista.append(layer1_concat)
    lista.append(layer1_sub)
    layer1_input = T.concatenate(lista,1) # [50 900]

    layer1_add = add(batch_size, alpha, beta, concat) # [50 300]

    lista = []
    lista.append(layer1_input)
    lista.append(layer1_add)

    return T.concatenate(lista,1) # [50 1200]

def mix4(layer1_inputs,batch_size,alpha,beta,concat):
    layer1_concat = concatenate_tensors(layer1_inputs) # [50 600]
    layer1_sub = sub(batch_size, 1, 1, concat) # [50 300]

    lista = []
    lista.append(layer1_concat)
    lista.append(layer1_sub)
    layer1_input = T.concatenate(lista,1) # [50 900]

    layer1_circ = circular_convolution(concat) # [50 300]

    lista = []
    lista.append(layer1_input)
    lista.append(layer1_circ)

    return T.concatenate(lista,1) # [50 1200] 