import numpy as np 
import time

def ThreeD_onehot(idx, depth):
    """
    for input with shape (timestep x batch x depth )
    """
    output_onehot =  np.zeros((idx.shape[0], idx.shape[1], depth))
    for timestep in range(len(idx)):
        for batch in range(len(idx[0])):
            max_idx=idx[timestep][batch]
            output_onehot[timestep][batch][max_idx]=1

    return output_onehot
class Sigmoid():
    def __init__(self,smooth=1):
        self.smooth = smooth
    def forward(self,inp_layer):
        self.inp_layer = inp_layer
        self.sigmoid = 1/(1+np.exp(-1*self.inp_layer*(1./self.smooth)))
        return self.sigmoid
    def backprop(self):
        return  self.sigmoid*(1-self.sigmoid)/self.smooth

class Tanh():
    def __init__(self,upperlimit=1,smooth=1):
        self.upperlimit = upperlimit
        self.smooth     = smooth
    def forward(self,inp_layer):
        self.inp_layer = inp_layer
        self.tanh =self.upperlimit*(np.exp(self.inp_layer*(2./self.smooth) )-1) / (np.exp(self.inp_layer*(2./self.smooth))+1)
        return self.tanh
    def backprop(self):
        return (1-self.tanh*self.tanh)/self.smooth

def square_loss(input_data,target):
    '''
    return L, dL
    '''
    batch = target.shape[0]
    L=  0.5*np.sum( (target-input_data)**2  )/batch
    All_dLoss = -target +input_data
    return  L, All_dLoss


def softmax_cross_entropy(input_data,target) :
    '''
    return prediction(softmax), L, dL
    '''
    after_softmax=[]
    batch = target.shape[0]
    # softmax
    for row in range(input_data.shape[0]) :
        this_row=np.exp(input_data[row])/np.sum(np.exp(input_data[row]))
        after_softmax.append(this_row)
    pred= np.array(after_softmax)
    # calculation of L
    small_num =np.zeros_like(target)
    small_num.fill(1e-8)   # prevent log(0)
    L=  -np.sum(np.multiply(target,np.log(pred+small_num) )  )/batch
    # calculation of dL
    All_dLoss = -target + pred
    return pred, L, All_dLoss

def timestep_softmax_cross_entropy(input_data,target) :
    '''
    input data : shape timestep x batch x depth
    return prediction(softmax), L, dL
    '''
    after_softmax=np.zeros_like(input_data)
    timesteps = input_data.shape[0]
    batch = input_data.shape[1]
    # softmax
    for timestep in range(timesteps):
        for single_data in range(batch) :
            this_row=np.exp(input_data[timestep][single_data])/np.sum(np.exp(input_data[timestep][single_data]))
            after_softmax[timestep][single_data]=this_row
    # calculation of L
    small_num =np.zeros_like(target)
    small_num.fill(1e-8)   # prevent log(0)
    L=  -np.sum(np.multiply(target,np.log(after_softmax+small_num) )  )/batch
    # calculation of dL
    All_dLoss = -target + after_softmax
    return after_softmax, L, All_dLoss

