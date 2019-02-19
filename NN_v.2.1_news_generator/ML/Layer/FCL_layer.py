import numpy as np
from ML.NN.Tools import orthogonal_initializer

class Full_Connected_Layer():
    def __init__(self, hidden_unit, ortho=False, dtype=np.float32):
        # hidden_unit : [input_hidden_units, output_hidden_units]
        self.W = (np.random.random(size=[hidden_unit[0],
                                         hidden_unit[1]])-0.5)/np.sqrt(hidden_unit[1])
        self.b = (np.random.random(size=[hidden_unit[1]])-0.5)/np.sqrt(hidden_unit[1])

        self.W = orthogonal_initializer(self.W) if ortho else self.W

        self.W = self.W.astype(dtype)
        self.b = self.b.astype(dtype)
        self.sum_dW = np.zeros_like(self.W)
        self.sum_db = np.zeros_like(self.b)
        self.dtype = dtype
        self.output = None
        self.timepiece = []

    def update(self, dW, db, lr):
        self.W = self.W - lr*dW
        self.b = self.b - lr*db

    def rewrite_Wb(self, W, b):
        self.W = W
        self.b = b

    def get_Wb(self):
        return self.W, self.b

    def get_dWb(self):
        return self.sum_dW, self.sum_db

class Full_Connected_Layer_NoBias():
    def __init__(self, hidden_unit, ortho=False, dtype=np.float32):
        # hidden_unit : [input_hidden_units, output_hidden_units]
        self.W = (np.random.random(size=[hidden_unit[0],
                                         hidden_unit[1]])-0.5)/np.sqrt(hidden_unit[1])
        self.W = self.W.astype(dtype)

        self.W = orthogonal_initializer(self.W) if ortho else self.W

        self.sum_dW = np.zeros_like(self.W)
        self.timepiece = []
        self.output = None

    def update(self, dW, db, lr):
        self.W = self.W -lr*dW

    def rewrite_Wb(self, W, b):
        self.W = W

    def get_Wb(self):
        return self.W, 0

    def get_dWb(self):
        return self.sum_dW, 0

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
        timesteps = x.shape[0]
        self.batch = x.shape[1]
        self.output_depth = x.shape[2]
        self.output = np.zeros((timesteps, self.batch, self.W.shape[1])).astype(self.dtype)
        for timestep in range(timesteps):
            self.output[timestep] = np.matmul(self.x[timestep], self.W)
        self.output = self.output + self.b
        return self.output

    def just_forward(self, x):
        this_output = np.matmul(x, self.W) + self.b
        return this_output

    def just_backprop(self, dLoss):
        return dLoss

    def timestep_forward(self, x):
        self.batch = x.shape[1]
        output = np.matmul(x, self.W) + self.b
        self.timepiece.append(x)
        return output

    def timestep_just_forward(self, x):
        self.batch = x.shape[1]
        output = np.matmul(x, self.W) + self.b
        self.timepiece.append(x)
        return output

    def timestep_gather(self):
        self.x = np.array(self.timepiece)
        self.timesteps = len(self.timepiece)
        self.timepiece = []

    def backprop(self, dLoss):
        # if use timestep (infer) apporach
        if len(self.timepiece) > 0:
            self.timestep_gather()
        self.x = self.x[:dLoss.shape[0]]
        dL_prev = np.zeros_like(self.x)
        self.sum_dW = np.zeros_like(self.W)
        for timestep in range(dLoss.shape[0]):
            dL_prev[timestep] = np.matmul(dLoss[timestep], self.W.T)
            self.sum_dW += np.dot(self.x[timestep].T, dLoss[timestep])
        self.sum_dW = self.sum_dW/dLoss.shape[1]
        self.sum_db = np.sum(np.sum(dLoss, axis=0), axis=0)/dLoss.shape[1]
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

    def timestep_gather(self):
        self.x = np.array(self.timepiece)
        self.x = self.x.reshape(len(self.timepiece), self.batch, -1)
        self.timesteps = len(self.timepiece)
        self.timepiece = []

    def backprop(self, dLoss):
        if len(self.timepiece) > 0:
            self.timestep_gather()
        self.x = self.x[:dLoss.shape[0]]
        dL_prev = np.zeros_like(self.x)
        self.sum_dW = np.zeros_like(self.W)
        for timestep in range(dLoss.shape[0]):
            dL_prev[timestep] = np.matmul(dLoss[timestep], self.W.T)
            self.sum_dW += np.dot(self.x[timestep].T, dLoss[timestep])
        self.sum_dW = self.sum_dW/dLoss.shape[1]
        return dL_prev

class partial_timestep_xW_b(Full_Connected_Layer):
    """
    for instance,
    input has 100 hidden units
    if only want to pass the last 30 hidden units into network.
    set hidden_unit(arguemnt) equals to (30, c)
    c can be any number, if want to maintain the same shape, set c equal to 30.
    input = [input_1(70 units), input_2(30 units)]
    output = np.concatenate( input_1, xW_b(input_2))
    """
    def __init__(self, hidden_unit, ortho=False, dtype=np.float32):
        # hidden_unit : [input_hidden_units, output_hidden_units]
        self.cut_at = hidden_unit[0]
        self.W = (np.random.random(size=[hidden_unit[0],
                                         hidden_unit[1]])-0.5)/np.sqrt(hidden_unit[1])
        self.b = (np.random.random(size=[hidden_unit[1]])-0.5)/np.sqrt(hidden_unit[1])

        self.W = orthogonal_initializer(self.W) if ortho else self.W

        self.W = self.W.astype(dtype)
        self.b = self.b.astype(dtype)

        self.sum_dW = np.zeros_like(self.W)
        self.sum_db = np.zeros_like(self.b)
        self.output = None
        self.timepiece = []

    def forward(self, x):
        self.x = x
        front_x = x[:, :, :-self.cut_at]
        back_x = x[:, :, -self.cut_at:]
        self.batch = x.shape[1]
        back_output = np.matmul(back_x, self.W) + self.b

        output = np.concatenate((front_x, back_output), axis=2)
        return output

    def timestep_forward(self, x):
        front_x = x[:, :, :-self.cut_at]
        back_x = x[:, :, -self.cut_at:]
        self.batch = x.shape[0]
        back_output = np.matmul(back_x, self.W) + self.b
        output = np.concatenate((front_x, back_output), axis=1)
        self.timepiece.append(x)
        return output

    def timestep_gather(self):
        self.x = np.array(self.timepiece)
        self.timesteps = len(self.timepiece)
        self.timepiece = []

    def backprop(self, dLoss):
        if len(self.timepiece) > 0:
            self.timestep_gather()
        self.x = self.x[:dLoss.shape[0]]
        back_x = self.x[:, :, -self.cut_at:]
        dL_prev = np.zeros_like(self.x)
        dL_prev[:, :, :-self.cut_at] = dLoss[:, :, :-self.cut_at]
        back_dLoss = dLoss[:, :, -self.cut_at:]
        self.sum_dW = np.zeros_like(self.W)
        for timestep in range(dLoss.shape[0]):
            dL_prev[timestep, :, -self.cut_at:] = np.matmul(back_dLoss[timestep], self.W.T)
            self.sum_dW += np.dot(back_x[timestep].T, back_dLoss[timestep])
        self.sum_dW = self.sum_dW/self.batch
        self.sum_db = np.sum(np.sum(back_dLoss, axis=0), axis=0)/self.batch
        return dL_prev


class annexed_timestep_xW_b(Full_Connected_Layer):
    """
    annexed_part = inp_layer*W +b
    return: (inp_layer, annexed_part)
    """
    def __init__(self, hidden_unit, ortho=False, dtype=np.float32):
        # hidden_unit : [input_hidden_units, output_hidden_units]
        self.annexed_depth = hidden_unit[1]
        self.W = (np.random.random(size=[hidden_unit[0],
                                         hidden_unit[1]])-0.5)/np.sqrt(hidden_unit[1])
        self.b = (np.random.random(size=[hidden_unit[1]])-0.5)/np.sqrt(hidden_unit[1])

        self.W = orthogonal_initializer(self.W) if ortho else self.W

        self.W = self.W.astype(dtype)
        self.b = self.b.astype(dtype)
        self.sum_dW = np.zeros_like(self.W)
        self.sum_db = np.zeros_like(self.b)
        self.output = None
        self.timepiece = []
        self.dtype = dtype

    def forward(self, x):
        self.x = x
        self.batch = x.shape[1]
        back_output = np.zeros((self.x.shape[0],
                                self.x.shape[1], self.W.shape[1])).astype(self.dtype)
        for timestep in range(self.x.shape[0]):
            back_output[timestep] = np.matmul(self.x[timestep], self.W)
        self.output = back_output + self.b

        output = np.concatenate((x, back_output), axis=2)
        return output

    def timestep_forward(self, x):
        self.batch = x.shape[0]
        back_output = np.matmul(x, self.W) + self.b
        output = np.concatenate((x, back_output), axis=1)
        self.timepiece.append(x)
        return output

    def timestep_gather(self):
        self.x = np.array(self.timepiece)
        self.timesteps = len(self.timepiece)
        self.timepiece = []

    def backprop(self, dLoss):
        if len(self.timepiece) > 0:
            self.timestep_gather()
        self.x = self.x[:dLoss.shape[0]]
        dL_prev = dLoss[:, :, :-self.annexed_depth]
        back_dLoss = dLoss[:, :, -self.annexed_depth:]
        self.sum_dW = np.zeros_like(self.W)
        for timestep in range(dLoss.shape[0]):
            dL_prev[timestep] += np.matmul(back_dLoss[timestep], self.W.T)
            self.sum_dW += np.dot(self.x[timestep].T, back_dLoss[timestep])
        self.sum_dW = self.sum_dW/self.batch
        self.sum_db = np.sum(np.sum(back_dLoss, axis=0), axis=0)/self.batch
        return dL_prev
class pre_multi_attn_layer(Full_Connected_Layer):
    """
    input -->  (value_1, value_2, value_3, value_4, key_1, key_2, key_3, key_4)
    e.g. for head 1:
    input*W1_value + b1_value = value_1
    input*W1_key   + b1_key   = key_1
    ... and so on.  
    return: (inp_layer, annexed_part)
    """
    def __init__(self, heads, value_matrix, key_matrix, ortho=False, dtype=np.float32):
        """
        heads: how many attention heads.
        value_matrix: (input_depth, value_matrix )
        key_martix: (input_depth, key_matrix )
        """
        self.heads = heads
        self.input_depth = value_matrix[0]
        self.value_depth = value_matrix[1]
        self.key_depth = key_matrix[1]
        self.head_depth = value_matrix[1] + key_matrix[1]
        self.total_depth = (value_matrix[1] + key_matrix[1])*self.heads
        self.W = (np.random.random(size=[self.input_depth, self.total_depth])-0.5)/np.sqrt(self.total_depth)
        self.b = (np.random.random(size=[self.total_depth])-0.5)/np.sqrt(self.total_depth)

        self.W = orthogonal_initializer(self.W) if ortho else self.W

        self.W = self.W.astype(dtype)
        self.b = self.b.astype(dtype)
        self.sum_dW = np.zeros_like(self.W)
        self.sum_db = np.zeros_like(self.b)
        self.output = None
        self.timepiece = []
        self.dtype = dtype

    def forward(self, x):
        self.x = x
        self.timesteps = x.shape[0]
        self.batch = x.shape[1]
        front_output = np.zeros((self.timesteps, self.batch, 
                                 self.value_depth*self.heads)).astype(self.dtype)
        back_output = np.zeros((self.timesteps, self.batch,
                                self.key_depth*self.heads)).astype(self.dtype)
        for timestep in range(self.timesteps):
            for this_head in range(self.heads):
                start_from = this_head*self.head_depth
                end_in = start_from + self.head_depth
                front_start = this_head*self.value_depth
                front_end = (1+this_head)*self.value_depth
                back_start = this_head*self.key_depth
                back_end = (1+this_head)*self.key_depth

                this_output = np.matmul(self.x[timestep], self.W[:, start_from:end_in])
                front_output[timestep, :, front_start:front_end] = this_output[:, :self.value_depth]
                back_output[timestep, :, back_start:back_end] = this_output[:, self.value_depth:]

        output = np.concatenate((front_output, back_output), axis=2) + self.b
        return output

    def timestep_forward(self, x):
        self.batch = x.shape[0]

        back_output = np.matmul(x, self.W[:, :self.value_depth])
        front_output= np.matmul(x, self.W[:, self.value_depth:])
        output = np.concatenate((x, back_output), axis=1) + self.b

        self.timepiece.append(x)
        return output

    def timestep_gather(self):
        self.x = np.array(self.timepiece)
        self.timesteps = len(self.timepiece)
        self.timepiece = []

    def backprop(self, dLoss):
        if len(self.timepiece) > 0:
            self.timestep_gather()
        dL_prev = np.zeros_like(self.x)
        front_dLoss = dLoss[:, :, :self.value_depth*self.heads]
        back_dLoss = dLoss[:, :, self.value_depth*self.heads:]
        self.sum_dW = np.zeros_like(self.W)
        for timestep in range(dLoss.shape[0]):
            for this_head in range(self.heads):
                start_from = this_head*self.head_depth
                end_in = start_from + self.head_depth
                front_start = this_head*self.value_depth
                front_end = (1+this_head)*self.value_depth
                back_start = this_head*self.key_depth
                back_end = (1+this_head)*self.key_depth

                this_front_dLoss = front_dLoss[timestep, :,front_start:front_end]
                this_back_dLoss = back_dLoss[timestep, :, back_start:back_end]
                this_dLoss = np.concatenate((this_front_dLoss, this_back_dLoss), axis=1)
                dL_prev[timestep] += np.matmul(this_dLoss, self.W[:, start_from:end_in].T)
                self.sum_dW[:, start_from:end_in] += np.dot(self.x[timestep].T, this_dLoss)

        self.sum_dW = self.sum_dW/self.batch
        self.sum_db = np.sum(np.sum(dLoss, axis=0), axis=0)/self.batch
        return dL_prev
 


class Embedding():
    "select a word vector, much more efficient than xW."
    def __init__(self, word2vector_array, trainable=False, dtype=np.float32):
        self.dtype = dtype
        self.W = word2vector_array
        self.vector_length = len(word2vector_array[0])
        self.b = None
        self.trainable = trainable
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

    def just_forward(self, x):
        return self.select(x)

    def timestep_forward(self, x):
        self.batch = x.shape[0]
        this_output = np.zeros((self.batch, self.vector_length)).astype(self.dtype)
        for batch in range(self.batch):
            this_output[batch] = self.W[np.argmax(x[batch])]
        return this_output

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
            self.sum_dW = np.zeros_like(self.W)
            for timestep in range(dLoss.shape[0]):
                self.sum_dW += np.dot(self.x.T, dLoss)
            self.sum_dW = self.sum_dW/(dLoss.shape[0]*dLoss.shape[1])
            dL_prev = np.zeros_like(self.x)
            for timestep in range(dLoss.shape[0]):
                dL_prev[timestep] = np.matmul(dLoss[timestep], self.W.T)
                self.sum_dW += np.dot(self.x.T, dLoss)
            self.sum_db = np.zeros_like(self.b)
            return dL_prev
        else:
            return dLoss

class softmax_cross_entropy():
    def __init__(self, mode='train'):
        self.dLoss = None
        self._dLoss = []
        self.Loss = 0
        self.mode = mode
        self._prediction = []
        self.use_timestep_forward = False
    def forward(self, input_data, target):
        """
        input_data, target in shape (T, B, D)
        """
        prediction = np.zeros_like(target)
        self.dLoss = np.zeros_like(target)
        for this_timestep in range(target.shape[0]):
            for this_batch in range(target.shape[1]):
                this_row = input_data[this_timestep][this_batch]
                exp_term = np.exp(this_row - np.max(this_row))
                softmax = exp_term/np.sum(exp_term)
                prediction[this_timestep][this_batch] = softmax
                self.dLoss[this_timestep][this_batch] = -target[this_timestep][this_batch]+softmax
        self.Loss = -np.sum(np.multiply(target, np.log(prediction+1e-6)))/target.shape[1]
        return prediction, self.Loss
    def timestep_forward(self, input_data, target):
        """
        input_data, target in shape (B, D)
        """
        if self.mode == 'train':
            self.use_timestep_forward = True
            prediction = np.zeros_like(input_data)
            this_dLoss = np.zeros_like(input_data)
            for this_batch in range(input_data.shape[0]):
                this_row = input_data[this_batch]
                exp_term = np.exp(this_row - np.max(this_row))
                softmax = exp_term/np.sum(exp_term)
                prediction[this_batch] = softmax
                this_dLoss[this_batch] = -target[this_batch]+softmax
            self._dLoss.append(this_dLoss)
            Loss = -np.sum(np.multiply(target, np.log(prediction+1e-6)))/target.shape[0]

        if self.mode == 'infer':
            self.use_timestep_forward = True
            prediction = np.zeros_like(input_data)
            for this_batch in range(input_data.shape[0]):
                this_row = input_data[this_batch]
                exp_term = np.exp(this_row - np.max(this_row))
                softmax = exp_term/np.sum(exp_term)
                prediction[this_batch] = softmax
                Loss = 0
            self._prediction.append(prediction)
        return prediction, Loss

    def backprop(self):
        if self.use_timestep_forward:
            self.timestep_gather()
        return self.dLoss

    def timestep_gather(self):
        self.prediction = np.array(self._prediction)
        self._prediction = []
        self.dLoss = np.array(self._dLoss)
        self._dLoss = []
        self.Loss = 0
    def get_pred(self):
        self.prediction = np.array(self._prediction)
        return self.prediction

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derv_sigmoid(x):
    return  sigmoid(x)*(1-sigmoid(x))

def softmax(x):
    after_softmax = []
    for row in range(x.shape[0]):
        this_row = np.exp(x[row])/np.sum(np.exp(x[row]))
        after_softmax.append(this_row)
    return np.array(after_softmax)
