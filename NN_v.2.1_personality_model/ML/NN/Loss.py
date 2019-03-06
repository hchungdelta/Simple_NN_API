import numpy as np
def square_loss(input_data, target):
    '''
    return L, dL
    '''
    batch = target.shape[0]
    L = 0.5* np.sum((target-input_data)**2)/batch
    All_dLoss = -target + input_data
    return  L, All_dLoss
def timestep_softmax_cross_entropy(input_data, target):
    '''
    input data : shape timestep x batch x depth
    return prediction(softmax), L, dL
    '''
    batch = input_data.shape[1]

    max_value = np.max(input_data, axis=2)
    tiled_max_value = np.einsum('ij,ijk->ijk', max_value, np.ones_like(input_data))
    rescale_exp_input = np.exp(input_data-tiled_max_value)
    sum_of_exp = 1/np.einsum('tbd->tb', rescale_exp_input)
    after_softmax = np.einsum('tbd,tb->tbd', rescale_exp_input, sum_of_exp)

    #tiled_max_valuhape(input_data.shape).transpose(0,2,1) calculation of L
    small_num = np.zeros_like(target)
    small_num.fill(1e-8)   # prevent log(0)
    L = -np.sum(np.multiply(target, np.log(after_softmax+small_num)))/batch
    # calculation of dL
    All_dLoss = -target + after_softmax
    return after_softmax, L, All_dLoss

def softmax_cross_entropy(input_data, target):
    '''
    return L, dL
    '''
    after_softmax = []
    batch = target.shape[0]
    # softmax
    for row in range(input_data.shape[0]):
        this_row = np.exp(input_data[row])/np.sum(np.exp(input_data[row]))
        after_softmax.append(this_row)
    pred = np.array(after_softmax)
    # calculation of L
    small_num = np.zeros_like(target)
    small_num.fill(1e-8)   # prevent log(0)
    L = -np.sum(np.multiply(target, np.log(pred+small_num)))/batch
    # calculation of dL
    All_dLoss = -target + pred
    # for single_data in range(batch) :
    #     dLoss = -target[single_data] +pred[single_data]
    #     All_dLoss.append(dLoss)
    # All_dLoss= np.array(All_dLoss)
    return pred, L, All_dLoss

def cross_entropy(target, prediction):
    '''
    return L, dL
    '''
    batch = target.shape[0]
    target = target
    pred = prediction
    small_num = np.zeros_like(target)
    small_num.fill(1e-8)   # prevent log(0)
    L = -np.sum(np.multiply(target, np.log(pred+small_num)))/batch

    All_dLoss = []
    for single_data in range(batch):
        dLoss = -target[single_data] +pred[single_data]
        All_dLoss.append(dLoss)
    All_dLoss = np.array(All_dLoss)
    return L, All_dLoss

def softmax(x):
    after_softmax = []
    for row in range(x.shape[0]):
        this_row = np.exp(x[row])/np.sum(np.exp(x[row]))
        after_softmax.append(this_row)
    return np.array(after_softmax)
