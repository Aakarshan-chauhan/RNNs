import numpy as np
def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


# Single LSTM cell    
def lstm(a_prev, c_prev, x_t, para):
    
    concat = np.concatenate((a_prev, x_t), axis = 0)
    
    # forget gate
    fg = sigmoid( para['Wf'].dot(concat) + para['bf'] )
    
    # input gate(update gate)
    ug = sigmoid( para['Wu'].dot(concat) + para['bu'] )
    
    # candidate value
    c_hat = np.tanh( para['Wc'].dot(concat) + para['bc'])
    
    # C t
    c_t = ug*c_hat + fg*c_prev
    
    # output gate
    og = sigmoid(para['Wo'].dot(concat) + para['bo'])
    
    # a T+1
    a_next = og*np.tanh(c_t)
    
    # Y t
    y_t= softmax(para['Wy'].dot(a_next) + para['by'])
    
    # cache for back propogation
    info = (a_prev, a_next, c_prev, c_next, x_t, para)
    
    return a_next, c_next, y_t, info


# Forward pass through LSTM with t_x timesteps
def lstm_forward(a0, X, para):
    
    nx, m, t_x = X.shape
    ny,na = para['Wy'].shape
    
    A = np.zeros((na, m, t_x))
    C = np.zeros((na, m, t_x))
    Y = np.zeros((ny, m, t_x))
    infos = []
    
    a_prev = a0
    c_prev = C[:, : ,0]
    
    for t in range(t_x):
        x_t = X[ :, :, t]
        a_next, c_next, y_t, cache = lstm(a_prev, c_prev, x_t, para)
        
        A[:, :, t] = a_next
        C[:, :, t] = c_next
        Y[:, :, t] = y_t
        infos.append(info)
        
        a_prev, c_prev = a_next, c_next
    
    infos = (infos, X)
    return A, C, Y, infos 

