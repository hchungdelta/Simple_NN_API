import numpy as np
import random 

def sigmoid(x):
    return (1/(1+np.exp(-x)))
def derv_sigmoid(x):
    return  sigmoid(x)*(1-sigmoid(x))

class dropout():
    def __init__ (self, keep_prob):
        self.keep_prob = keep_prob
    def forward(self, input_data) :
        # layer shape must be( batch,depth)
        # make some neurals equal to zero
        # all batch have the same muted neural
        dropout_layer = np.zeros_like(input_data)
        self.alive_neural = random.sample(range(input_data.shape[1]), int(input_data.shape[1]*self.keep_prob))
        for neural in self.alive_neural:
            for batch in range(input_data.shape[0]):
                dropout_layer[batch][neural] = input_data[batch][neural]
        return dropout_layer

    def backprop(self, dLoss):
        dL = np.zeros_like(dLoss)
        for neural in self.alive_neural:
            for batch in range(dLoss.shape[0]):
                dL[batch][neural] = dLoss[batch][neural]
        return dL

class Full_Connected_Layer():
    def __init__(self, hidden_unit):
        # hidden_unit : [input_hidden_units, output_hidden_units]
        self.W = np.random.normal(size=[hidden_unit[0], hidden_unit[1]])
        self.b = np.random.normal(size=[hidden_unit[1]])
    def update(self, dW, db, lr):
        self.W = self.W -lr*dW
        self.b = self.b -lr*db
    def dropout(self, keep_prob) :
        # layer shape must be( batch,depth)
        # make some neurals equal to zero
        # all batch have the same muted neural
        dropout_layer = np.zeros_like(self.output)
        alive_neural = random.sample(range(self.output.shape[1]), int(self.output.shape[1]*keep_prob))
        for neural in alive_neural:
            for batch in range(self.output.shape[0]):
                dropout_layer[batch][neural] = self.output[batch][neural]
        self.output = dropout_layer
        return self.output
    def rewrite_Wb(self, W, b):
        self.W = W
        self.b = b
    def get_Wb(self):
        return self.W, self.b
    def get_dWb(self):
        return self.sum_dW, self.sum_db

class sigmoid_xW_b(Full_Connected_Layer):
    def forward(self, x):
        self.x = x # input
        self.batch = x.shape[0]
        self.output_depth = x.shape[1]
        z = np.matmul(self.x, self.W) + np.tile(self.b, (self.batch, 1))
        self.output = sigmoid(z)
        return self.output
    def backprop(self, dLoss):
        sum_db = np.zeros_like(self.output_depth)
        sum_dW = np.zeros_like(self.W)
        dL_prev = np.zeros_like(self.x)
        for single_data in range(self.batch):
            dL = np.multiply(dLoss[single_data], derv_sigmoid(self.output[single_data]))
            db = dL
            dW = np.outer(self.x[single_data].T, dL)
            sum_db = np.add(sum_db, db)
            sum_dW = np.add(sum_dW, dW)
            dL_prev[single_data] = np.matmul(dL, self.W.T)
        # divided by batch 
        self.sum_db = sum_db*(1./self.batch)
        self.sum_dW = sum_dW*(1./self.batch)
        return dL_prev

class xW_b(Full_Connected_Layer):
    def forward(self, x):
        self.x = x # input
        self.batch = x.shape[0]
        self.output_depth = x.shape[1]
        self.output = np.matmul(self.x, self.W) + np.tile(self.b, (self.batch, 1))# self.b
        return self.output
    def backprop(self, dLoss):
        sum_db = np.zeros_like(self.output_depth)
        sum_dW = np.zeros_like(self.W)
        dL_prev = np.zeros_like(self.x)
        for single_data in range(self.batch):       
            dL = dLoss[single_data]
            db = dL
            dW = np.outer(self.x[single_data].T, dL)
            sum_db = np.add(sum_db,db)
            sum_dW = np.add(sum_dW,dW)
            dL_prev[single_data] = np.matmul(dL, self.W.T)
        # divided by batch
        self.sum_db = sum_db*(1./self.batch)
        self.sum_dW = sum_dW*(1./self.batch)
        return dL_prev



def softmax_cross_entropy(input_data, target):
    '''
    return softmax(input_data, L, dL
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
    L= -np.sum(np.multiply(target, np.log(pred+small_num)))/batch
    # calculation of dL
    All_dLoss = []
    for single_data in range(batch):
        dLoss = -target[single_data] + pred[single_data]
        All_dLoss.append(dLoss)
    All_dLoss = np.array(All_dLoss)
    return pred, L,All_dLoss

def cross_entropy(target,prediction):
    '''
    return L, dL
    '''
    batch = target.shape[0]
    target= target 
    pred  = prediction
    small_num =np.zeros_like(target)
    small_num.fill(1e-8)   # prevent log(0)  
    L=  -np.sum(np.multiply(target,np.log(pred+small_num) )  )/batch

    All_dLoss=[]
    for single_data in range(batch) :
        dLoss = -target[single_data] +pred[single_data]
        All_dLoss.append(dLoss)
    All_dLoss= np.array(All_dLoss)
    return L, All_dLoss

def softmax(x):
    after_softmax=[]
    for row in range(x.shape[0]) :
        this_row=np.exp(x[row])/np.sum(np.exp(x[row]))
        after_softmax.append(this_row) 
    return np.array(after_softmax)

def initialize_layer(input_unit,output_unit):
    W=np.random.normal(size=[input_unit,output_unit])
    b=np.random.normal(size=[output_unit])
    return W,b

def accuracy_test(pred,target):
    accuracy = 0 
    for element in range(len(pred)):
        if pred[element] == target[element] :
            accuracy += 1./len(pred)
    return accuracy

def onehot(number,depth):
    array=np.zeros(depth,dtype='f')
    array[number]=1.
    return array

