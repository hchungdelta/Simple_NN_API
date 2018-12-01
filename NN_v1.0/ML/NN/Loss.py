import numpy as np
import random 

def softmax_cross_entropy(input_data,target):
    '''
    return softmax(input_data), L, dL
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
    All_dLoss = []
    for single_data in range(batch):
        dLoss = -target[single_data] + pred[single_data]
        All_dLoss.append(dLoss)
    All_dLoss = np.array(All_dLoss)
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
    L = -np.sum(np.multiply(target,np.log(pred+small_num)))/batch

    All_dLoss=[]
    for single_data in range(batch):
        dLoss = -target[single_data] + pred[single_data]
        All_dLoss.append(dLoss)
    All_dLoss = np.array(All_dLoss)
    return L, All_dLoss

def softmax(x):
    after_softmax = []
    for row in range(x.shape[0]):
        this_row = np.exp(x[row])/np.sum(np.exp(x[row]))
        after_softmax.append(this_row) 
    return np.array(after_softmax)

def accuracy_test(pred, target):
    accuracy = 0 
    for element in range(len(pred)):
        if pred[element] == target[element]:
            accuracy += 1./len(pred)
    return accuracy


