"""
Conv2conv layers. (1D)
For SandGlass model use.
"""
import numpy as np
from numba import jit
from ML.Layer.Acti_layer import Sigmoid, Tanh
from ML.NN.Tools import orthogonal_initializer

class trainable_layer:

    def update(self, dW, db, lr):
        self.W = self.W - dW * lr
        self.b = self.b - db * lr

    def get_dWb(self):
        return self.dW, self.db

    def get_Wb(self):
        return self.W, self.b

    def rewrite_Wb(self, W, b):
        self.W = W
        self.b = b

class Conv1D(trainable_layer):
    def __init__(self, hidden_units, ortho=False, stride=1, residual=False, dtype=np.float32):
        """
        hidden_units : in shape [input_filter,output_filter,kernel_size,input_depth,output_depth]
        stride       : displacement of one step.
        residual     : output = output + input  (short connection)
        ortho        : Orthogonal matrices
        """
        self.dtype = dtype
        self.kernel_size = hidden_units[0]
        self.input_depth = hidden_units[1]
        self.output_depth = hidden_units[2]
        normalization = self.kernel_size*self.input_depth
        self.W = (np.random.random(hidden_units)-0.5)/np.sqrt(normalization)
        self.b = (np.random.random((self.output_depth))-0.5)/np.sqrt(normalization)

        self.W = orthogonal_initializer(self.W) if ortho else self.W

        self.W = self.W.astype(self.dtype)
        self.b = self.b.astype(self.dtype)

        self.amount_of_pad = 0
        self.stride = stride
        self.residual = residual

    @jit(fastmath=True)
    def forward(self, inp):
        self.inp = inp
        self.total_step = self.inp.shape[0]
        self.batch = self.inp.shape[1]
        self.pad_inp = self.pad_this_inp()
        self.pad_total_step = self.pad_inp.shape[0]
        output = np.zeros((self.total_step, self.batch, self.output_depth)).astype(self.dtype)

        for this_step in range(self.total_step):
            for k_idx in range(self.kernel_size):
                weighted_pad_inp = np.dot(self.pad_inp[this_step+k_idx],
                                          self.W[k_idx])
                output[this_step] += weighted_pad_inp 
            output[this_step] += self.b
        if self.residual:
            output = output + self.inp
        return output

    @jit(fastmath=True)
    def backprop(self, dLoss):
        dL_prev = np.zeros_like(self.pad_inp)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.pad_inp_timestep = self.pad_inp.shape[0]
        normalization = self.total_step*self.batch
        self.db = np.einsum('tbd->d', dLoss)/normalization
        for idx in range(self.kernel_size):
            for this_out, this_inp in enumerate(
                    range(idx, self.pad_inp_timestep-(self.kernel_size-(idx+1)))):
                this_x = self.pad_inp[this_inp]
                self.dW[idx] += np.dot(this_x.T, dLoss[this_out])/normalization

        for idx in range(self.total_step):
            for k_idx in range(self.kernel_size):
                dL_prev[idx+k_idx] += np.dot(dLoss[idx], self.W[k_idx].T)
        dL_prev = dL_prev[self.amount_of_pad:]
        if self.residual:
            dL_prev = dL_prev + dLoss
        return dL_prev


    def pad_this_inp(self):
        self.amount_of_pad = self.kernel_size - 1
        # if even, add equal amount on both sides, if odds, add more one pad at the front.
        pads_front = np.tile(np.zeros((self.batch, self.input_depth)).astype(self.dtype),
                             (self.amount_of_pad, 1, 1))
        return  np.concatenate((pads_front, self.inp), axis=0)


class conv1D_group():
    """
    common functions of conv1Ds and conv1Ds_rev
    """
    def __init__(self, conv1Ds_list):
        self.conv1Ds = conv1Ds_list
        self.amount_of_conv1Ds = len(conv1Ds_list)

    def update(self, dWs, dbs, lr):
        for idx, this_conv1D in enumerate(self.conv1Ds):
            this_conv1D.update(dWs[0][idx], dbs[0][idx], lr)

    def get_dWb(self):
        dWs = []
        dbs = []
        for idx, this_conv1D in enumerate(self.conv1Ds):
            this_dW, this_db = this_conv1D.get_dWb()
            dWs.append(this_dW)
            dbs.append(this_db)
        return dWs, dbs

    def get_Wb(self):
        Ws = []
        bs = []
        for idx, this_conv1D in enumerate(self.conv1Ds):
            this_W, this_b = this_conv1D.get_Wb()
            Ws.append(this_W)
            bs.append(this_b)
        return Ws, bs

    def rewrite_Wb(self, Ws, bs):
        for idx, this_conv1D in enumerate(self.conv1Ds):
            this_conv1D.rewrite_Wb(Ws[idx], bs[idx])


