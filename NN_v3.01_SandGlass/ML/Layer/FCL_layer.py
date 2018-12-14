import random
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
def derv_sigmoid(x):
    return  sigmoid(x)*(1-sigmoid(x))

class dropout_4d():
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
    def forward(self, input_data):
        dropout_layer = np.zeros_like(input_data)
        self.alive_neural = random.sample(range(input_data.shape[3]),
                                          int(input_data.shape[3]*self.keep_prob))
        for neural in self.alive_neural:
            dropout_layer[:, :, :, neural] = input_data[:, :, :, neural]
        return dropout_layer

    def backprop(self, dLoss):
        dL = np.zeros_like(dLoss)
        for neural in self.alive_neural:
            dL[:, :, :, neural] = dLoss[:, :, :, neural]
        return dL



class dropout_3d():
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
    def forward(self, input_data):
        dropout_layer = np.zeros_like(input_data)
        self.alive_neural = random.sample(range(input_data.shape[2]),
                                          int(input_data.shape[2]*self.keep_prob))
        for neural in self.alive_neural:
            dropout_layer[:, :, neural] = input_data[:, :, neural]
        return dropout_layer

    def backprop(self, dLoss):
        dL = np.zeros_like(dLoss)
        for neural in self.alive_neural:
            dL[:, :, neural] = dLoss[:, :, neural]
        return dL

class dropout_2d():
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
    def forward(self, input_data):
        dropout_layer = np.zeros_like(input_data)
        self.alive_neural = random.sample(range(input_data.shape[1]),
                                          int(input_data.shape[1]*self.keep_prob))
        for neural in self.alive_neural:
            dropout_layer[:, neural] = input_data[:, neural]
        return dropout_layer

    def backprop(self, dLoss):
        dL = np.zeros_like(dLoss)
        for neural in self.alive_neural:
            dL[:, neural] = dLoss[:, neural]
        return dL



class dropout():
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
    def forward(self, input_data):
        dropout_layer = np.zeros_like(input_data)
        self.alive_neural = random.sample(range(input_data.shape[1]),
                                          int(input_data.shape[1]*self.keep_prob))
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
    def __init__(self, hidden_unit, random=False, dtype=np.float32):
        # hidden_unit : [input_hidden_units, output_hidden_units]
        self.W = (np.random.random(size=[hidden_unit[0],
                                         hidden_unit[1]])-0.5)/np.sqrt(hidden_unit[1])
        self.b = (np.random.random(size=[hidden_unit[1]])-0.5)/np.sqrt(hidden_unit[1])
        self.W = self.W.astype(dtype)
        self.b = self.b.astype(dtype)
        self.sum_dW = np.zeros_like(self.W)
        self.sum_db = np.zeros_like(self.b)
        self.output = None
        self.timepiece = []
        self.random = random
    def update(self, dW, db, lr):
        self.W = self.W - lr*dW
        self.b = self.b - lr*db
    def dropout(self, keep_prob):
        dropout_layer = np.zeros_like(self.output)
        alive_neural = random.sample(range(self.output.shape[1]),
                                     int(self.output.shape[1]*keep_prob))
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

class Full_Connected_Layer_NoBias():
    def __init__(self, hidden_unit, dtype=np.float32):
        # hidden_unit : [input_hidden_units, output_hidden_units]
        self.W = (np.random.random(size=[hidden_unit[0],
                                         hidden_unit[1]])-0.5)/np.sqrt(hidden_unit[1])
        self.W = self.W.astype(dtype)
        self.sum_dW = np.zeros_like(self.W)
        self.timepiece = []
        self.output = None
    def update(self, dW, db, lr):
        self.W = self.W -lr*dW
    def dropout(self, keep_prob):
        dropout_layer = np.zeros_like(self.output)
        alive_neural = random.sample(range(self.output.shape[1]),
                                     int(self.output.shape[1]*keep_prob))
        for neural in alive_neural:
            for batch in range(self.output.shape[0]):
                dropout_layer[batch][neural] = self.output[batch][neural]
        self.output = dropout_layer
        return self.output
    def rewrite_Wb(self, W, b):
        self.W = W
    def get_Wb(self):
        return self.W, 0
    def get_dWb(self):
        return self.sum_dW, 0

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
        self.output = np.matmul(self.x, self.W) + self.b #  np.tile(self.b,(self.batch,1))# self.b
        return self.output
    def backprop(self, dLoss):
        dL_prev = np.einsum('bo,io->bi', dLoss, self.W)
        sum_db = np.sum(dLoss, axis=0)
        sum_dW = np.einsum('bi,bo->io', self.x, dLoss)
        self.sum_db = sum_db*(1./self.batch)
        self.sum_dW = sum_dW*(1./self.batch)
        return dL_prev

class timestep_xW_b(Full_Connected_Layer):
    def forward(self, x):
        self.x = x # input
        self.timesteps = x.shape[0]
        self.batch = x.shape[1]
        self.output_depth = x.shape[2]
        self.output = np.matmul(self.x, self.W) + np.tile(self.b, (self.timesteps, self.batch, 1))
        return self.output
    def just_forward(self, x):
        this_timesteps = x.shape[0]
        this_batch = x.shape[1]
        this_output = np.matmul(x, self.W) + np.tile(self.b, (this_timesteps, this_batch, 1))
        return this_output

    def timepiece_forward(self, x):
        self.batch = x.shape[1]
        output = np.matmul(x, self.W) + np.tile(self.b, (1, self.batch, 1))
        self.timepiece.append(x)
        return output
    def timepiece_gather(self):
        self.x = np.array(self.timepiece)
        self.x = self.x.reshape(len(self.timepiece), self.batch, -1)
        self.timesteps = len(self.timepiece)
        self.timepiece = []
    def backprop(self, dLoss):
        # cut out for max length
        self.x = self.x[:dLoss.shape[0]]

        if not self.random:
            self.sum_dW = np.einsum('tij,tik->jk', self.x, dLoss)
            self.sum_dW = self.sum_dW/(dLoss.shape[0]*dLoss.shape[1])

        if self.random:
            # randomly choose one batch for training output dense layer
            # just a dirty & rough alternative for sparse softmax.
            random_one_in_batch = np.random.randint(self.batch)
            temp_x = self.x[:, random_one_in_batch, :]
            temp_dLoss = dLoss[:, random_one_in_batch, :]
            self.sum_dW = np.einsum('tj,tk->jk', temp_x, temp_dLoss)

        dL_prev = np.zeros_like(self.x)
        for timestep in range(dLoss.shape[0]):
            dL_prev[timestep] = np.matmul(dLoss[timestep], self.W.T)

        self.sum_db = np.sum(np.sum(dLoss, axis=0), axis=0)/(dLoss.shape[0]*dLoss.shape[1])
        return dL_prev

class timestep_xW(Full_Connected_Layer_NoBias):
    def forward(self, x):
        self.x = x # input
        self.timesteps = x.shape[0]
        self.batch = x.shape[1]
        self.output_depth = x.shape[2]
        # in some cases, for can be faster than huge tensor ( even x10 faster)
        self.output = np.zeros((self.timesteps, self.batch, self.W.shape[1]))
        for timestep in range(self.timesteps):
            self.output[timestep] = np.matmul(self.x[timestep], self.W)
        return self.output
    def just_forward(self, x):
        this_output = np.matmul(x, self.W)
        return this_output

    def timepiece_forward(self, x):
        self.batch = x.shape[1]
        output = np.matmul(x, self.W)
        self.timepiece.append(x)
        return output
    def timepiece_gather(self):
        self.x = np.array(self.timepiece)
        self.x = self.x.reshape(len(self.timepiece), self.batch, -1)
        self.timesteps = len(self.timepiece)
        self.timepiece = []

    def backprop(self, dLoss, random=False):
        # cut out for max length
        self.x = self.x[:dLoss.shape[0]]

        if not random:
            self.sum_dW = np.einsum('tij,tik->jk', self.x, dLoss)
            self.sum_dW = self.sum_dW/(dLoss.shape[0]*dLoss.shape[1])

        if random:
            # randomly choose one batch for training output dense layer
            # just a dirty & rough alternative for sparse softmax.
            random_one_in_batch = np.random.randint(self.batch)
            temp_x = self.x[:, random_one_in_batch, :]
            temp_dLoss = dLoss[:, random_one_in_batch, :]
            self.sum_dW = np.einsum('tj,tk->jk', temp_x, temp_dLoss)

        dL_prev = np.zeros_like(self.x)
        for timestep in range(dLoss.shape[0]):
            dL_prev[timestep] = np.matmul(dLoss[timestep], self.W.T)

        return dL_prev


class Embedding():
    "need to modify the code to trainable / non-trainable"
    def __init__(self, word2vector_array, trainable=False, random=False, dtype=np.float32):
        self.dtype = dtype
        self.W = word2vector_array
        self.vector_length = len(word2vector_array[0])
        self.b = None
        self.trainable = trainable
        self.random = random
        if self.trainable:
            def get_dWb(self):
                return self.sum_dW, None
            def update(self, dW, db, lr):
                self.W = self.W -lr*dW
    def rewrite_Wb(self, W, b):
        self.W = W
        self.b = None
    def get_Wb(self):
        return self.W, self.b
    def forward(self, x):
        return self.select(x)
    def select(self, x):
        self.timestep = x.shape[0]
        self.batch = x.shape[1]
        this_output = np.zeros((self.timestep, self.batch, self.vector_length)).astype(self.dtype)
        for timestep in range(self.timestep):
            for batch in range(self.batch):
                this_output[timestep][batch] = self.W[np.argmax(x[timestep][batch])]
        return this_output
    def backprop(self, dLoss):
        if self.trainable:
            self.x = self.x[:dLoss.shape[0]]
            if not self.random:
                self.sum_dW = np.einsum('tij,tik->jk', self.x, dLoss)
                self.sum_dW = self.sum_dW/(dLoss.shape[0]*dLoss.shape[1])
            if self.random:
                random_one_in_batch = np.random.randint(self.batch)
                temp_x = self.x[:, random_one_in_batch, :]
                temp_dLoss = dLoss[:, random_one_in_batch, :]
                self.sum_dW = np.einsum('tj,tk->jk', temp_x, temp_dLoss)

            dL_prev = np.zeros_like(self.x)
            for timestep in range(dLoss.shape[0]):
                dL_prev[timestep] = np.matmul(dLoss[timestep], self.W.T)
            self.sum_db = np.zeros_like(self.b)
            return dL_prev
        else:
            return dLoss


def softmax_cross_entropy(input_data, target):
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
        dLoss = -target[single_data] +pred[single_data]
        All_dLoss.append(dLoss)
    All_dLoss = np.array(All_dLoss)
    return pred, L, All_dLoss

def cross_entropy(target, prediction):
    batch = target.shape[0]
    target = target
    pred = prediction
    small_num = np.zeros_like(target)
    small_num.fill(1e-8)   # prevent log(0)
    L = -np.sum(np.multiply(target, np.log(pred+small_num)))/batch

    All_dLoss = []
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

def onehot(number, depth):
    array = np.zeros(depth, dtype='f')
    array[number] = 1.
    return array
