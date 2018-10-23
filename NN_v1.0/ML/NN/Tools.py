import numpy as np

def softmax(x):
    after_softmax=[]
    for row in range(x.shape[0]) :
        this_row=np.exp(x[row])/np.sum(np.exp(x[row]))
        after_softmax.append(this_row) 
    return np.array(after_softmax)

def accuracy_test(pred,target):
    accuracy = 0 
    for element in range(len(pred)):
        if pred[element] == target[element] :
            accuracy += 1./len(pred)
    return accuracy


