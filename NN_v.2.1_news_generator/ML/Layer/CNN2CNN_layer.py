"""
Conv2conv layers. (1D)
For SandGlass model use.
"""
import numpy as np
from numba import jit
from ML.Layer.Acti_layer import Sigmoid, Tanh

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


class conv1D(trainable_layer):
    def __init__(self, hidden_units, paddling=True, stride=1, residual=False, dtype=np.float32):
        """
        hidden_units : in shape [input_filter,output_filter,kernel_size,input_depth,output_depth]
        padding      : add zero paddlings at both sides of input.
        stride       : displacement of one step.
        residual     : output = output + input  (short connection)
        ortho        : Orthogonal matrices
        """
        self.dtype = dtype
        self.input_filters = hidden_units[0]
        self.output_filters = hidden_units[1]
        self.kernel_size = hidden_units[2]
        self.input_depth = hidden_units[3]
        self.output_depth = hidden_units[4]
        normalization = self.input_filters*self.output_filters*self.kernel_size*self.input_depth
        self.W = (np.random.random(hidden_units)-0.5)/np.sqrt(normalization)
        self.b = (np.random.random((self.input_filters,
                                    self.output_filters,
                                    self.output_depth))-0.5)/np.sqrt(normalization)


        self.W = self.W.astype(self.dtype)
        self.b = self.b.astype(self.dtype)

        self.paddling = paddling
        self.stride = stride
        self.residual = residual

    @jit(fastmath=True)
    def forward(self, inp):
        self.inp = inp
        self.total_step = self.inp.shape[1]
        self.batch = self.inp.shape[2]
        self.pad_inp = self.pad_this_inp()
        self.pad_total_step = self.pad_inp.shape[1]
        output = np.zeros((self.output_filters,
                           self.total_step, self.batch, self.output_depth)).astype(self.dtype)

        for this_input_filter in range(self.input_filters):
            for this_output_filter in range(self.output_filters):
                for this_step in range(self.total_step):
                    for k_idx in range(self.kernel_size):
                        weighted_pad_inp = np.dot(self.pad_inp[this_input_filter, this_step+k_idx],
                                                  self.W[this_input_filter, this_output_filter, k_idx])
                        output[this_output_filter, this_step] += weighted_pad_inp
                    output[this_output_filter, this_step] += self.b[this_input_filter,
                                                                    this_output_filter]
        if self.residual:
            output = output + self.inp
        return output

    @jit(fastmath=True)
    def backprop(self, dLoss):
        dL_prev = np.zeros_like(self.pad_inp)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.pad_inp_timestep = self.pad_inp.shape[1]
        normalization = self.total_step*self.batch*self.output_filters
        for this_input_filter in range(self.input_filters):
            for this_output_filter in range(self.output_filters):
                self.db[this_input_filter, this_output_filter] = np.einsum(
                    'tbd->d', dLoss[this_output_filter])/normalization

                for idx in range(self.kernel_size):
                    for this_out, this_inp in enumerate(
                            range(idx, self.pad_inp_timestep-(self.kernel_size-(idx+1)))):
                        this_x = self.pad_inp[this_input_filter, this_inp]
                        self.dW[this_input_filter, this_output_filter, idx] += np.dot(
                            this_x.T, dLoss[this_output_filter, this_out])/normalization

                for idx in range(self.total_step):
                    for k_idx in range(self.kernel_size):
                        dL_prev[this_input_filter, idx+k_idx] += np.dot(
                            dLoss[this_output_filter, idx],
                            self.W[this_input_filter, this_output_filter, k_idx].T)

        dL_prev = dL_prev[:, self.amount_of_pad_front:self.pad_total_step-self.amount_of_pad_end]
        if self.residual:
            dL_prev = dL_prev + dLoss
        return dL_prev


    def pad_this_inp(self):
        amount_of_pad = self.kernel_size - 1
        # if even, add equal amount on both sides, if odds, add more one pad at the front.
        self.amount_of_pad_front = amount_of_pad //2  + amount_of_pad % 2
        self.amount_of_pad_end = amount_of_pad //2

        pads_front = np.tile(np.zeros((self.batch, self.input_depth)).astype(self.dtype),
                             (self.input_filters, self.amount_of_pad_front, 1, 1))
        pads_end = np.tile(np.zeros((self.batch, self.input_depth)).astype(self.dtype),
                           (self.input_filters, self.amount_of_pad_end, 1, 1))
        return  np.concatenate((pads_front, self.inp, pads_end), axis=1)


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


class conv1Ds(conv1D_group):
    """
    hold a number of conv1D layer.
    """
    def forward(self, inp):
        self.inp = inp
        self.inp_conv1D_amount = inp.shape[0]
        concat_output = []
        for idx, this_conv1D in enumerate(self.conv1Ds):
            concat_output.append(this_conv1D.forward(inp))
        return np.squeeze(np.array(concat_output), 1)

    def backprop(self, dLoss):
        dL_prev = np.zeros_like(self.inp)
        for idx, this_conv1D in enumerate(self.conv1Ds):
            this_dLoss = np.expand_dims(dLoss[idx], 0)
            dL_prev += this_conv1D.backprop(this_dLoss)
        return  dL_prev


class conv1Ds_rev(conv1D_group):
    """
    hold a number of conv1D_rev layer.
    """
    def forward(self, inp):
        self.inp = inp
        self.inp_conv1D_amount = inp.shape[0]
        sum_output = []
        for idx, this_conv1D in enumerate(self.conv1Ds):
            sum_output.append(this_conv1D.forward(np.array([inp[idx]])))
        return np.sum(sum_output, axis=0)

    def backprop(self, dLoss):
        dL_prev = np.zeros_like(self.inp)
        for idx, this_conv1D in enumerate(self.conv1Ds):
            dL_prev[idx] = this_conv1D.backprop(dLoss)
        return  dL_prev





class ReduceAttn():
    def __init__(self, reduce_size, paddling=True, dtype=np.float32):
        """
        reduce_size  : input_timestep -> input_timestep/reduce_size
        padding      : add zero paddlings at both sides of input.

        """
        self.dtype = dtype
        self.reduce_size = reduce_size
        self.paddling = paddling

    @jit(fastmath=True)
    def forward(self, inp):
        self.inp = inp
        self.total_step = self.inp.shape[1]
        self.batch = self.inp.shape[2]
        self.input_depth = self.inp.shape[3]
        self.pad_inp = self.pad_this_inp()
        self.pad_total_step = self.pad_inp.shape[1]
        self.reduce_total_step = int(self.pad_total_step/self.reduce_size)
        self.alpha = np.zeros((self.reduce_total_step,
                               self.batch, self.reduce_size)).astype(self.dtype)
        output = np.zeros((1, self.reduce_total_step,
                           self.batch, self.input_depth)).astype(self.dtype)
        for out_idx in range(self.reduce_total_step):
            this_inp_group = self.pad_inp[0, out_idx*self.reduce_size:(1+out_idx)*self.reduce_size]
            _score = np.einsum('Kbd,Gbd->bK', this_inp_group, this_inp_group)
            contrib_fromself = np.einsum('Kbd,Kbd->bK', this_inp_group, this_inp_group)
            score = _score - contrib_fromself
            max_score = np.tile(np.max(score, axis=1), (self.reduce_size, 1)).transpose(1, 0)
            score = np.exp(score - max_score)
            sum_score = 1/np.sum(score, axis=1)
            this_alpha = np.einsum('bK,b->bK', score, sum_score)
            self.alpha[out_idx] = this_alpha
            output[0, out_idx] = np.einsum('bK,Kbd->bd', this_alpha, this_inp_group)

        return output

    def backprop(self, dLoss):
        dL_prev = np.zeros_like(self.pad_inp)
        for this_timestep in range(self.reduce_total_step):
            this_dL_prev = np.einsum('bd,bK->Kbd',
                                     dLoss[0, this_timestep],
                                     self.alpha[this_timestep])
            dL_prev[0, this_timestep*self.reduce_size:(this_timestep+1)*self.reduce_size] += this_dL_prev
            this_inp_group = self.pad_inp[0, this_timestep*self.reduce_size:(1+this_timestep)*self.reduce_size]
            sum_this_inp_group = np.einsum('Kbd->bd', this_inp_group)

            dalpha_score = -1*np.einsum('bK,bk->bKk',
                                        self.alpha[this_timestep],
                                        self.alpha[this_timestep])
            dalpha_score += np.einsum('bK,Kk->bKk',
                                      self.alpha[this_timestep],
                                      np.eye(self.reduce_size))
            for kernel_idx in range(self.reduce_size):
                dscore_hidden = np.copy(this_inp_group)
                dscore_hidden[kernel_idx] += sum_this_inp_group - 2*dscore_hidden[kernel_idx]
                cor = np.einsum('bKk,kbd,Kbd->bd', dalpha_score, dscore_hidden, this_inp_group)
                dL_prev[0, this_timestep*self.reduce_size+kernel_idx] += dLoss[0, this_timestep]*cor
        dL_prev = dL_prev[:, self.amount_of_pad_front:self.total_step-self.amount_of_pad_end]
        return dL_prev

    def pad_this_inp(self):
        amount_of_pad = self.inp.shape[1] % self.reduce_size
        # if even, add equal amount on both sides, if odds, add more one pad at the front.
        self.amount_of_pad_front = amount_of_pad //2 + amount_of_pad%2
        self.amount_of_pad_end = amount_of_pad //2

        pads_front = np.tile(np.zeros((self.batch, self.input_depth)).astype(self.dtype),
                             (1, self.amount_of_pad_front, 1, 1))
        pads_end = np.tile(np.zeros((self.batch, self.input_depth)).astype(self.dtype),
                           (1, self.amount_of_pad_end, 1, 1))
        return  np.concatenate((pads_front, self.inp, pads_end), axis=1)

    def give_me_alpha(self):
        return self.alpha



class ReduceConv(trainable_layer):
    def __init__(self, hidden_units, paddling=True, dtype=np.float32):
        """
        hidden_units : in shape (reduce_size, input_depth,output_depth)
        reduce_size  : input_timestep -> input_timestep/reduce_size
        padding      : add zero paddlings at both sides of input.
        """
        self.dtype = dtype
        self.reduce_size = hidden_units[0]
        self.input_depth = hidden_units[1]
        self.output_depth = hidden_units[2]
        self.W = (np.random.random(hidden_units)-0.5) /np.sqrt(self.output_depth)
        self.b = (np.random.random((self.output_depth))-0.5)/np.sqrt(self.output_depth)




        self.W = self.W.astype(self.dtype)
        self.b = self.b.astype(self.dtype)
        self.paddling = paddling

    @jit(fastmath=True)
    def forward(self, inp):
        """
        inp : input data in shape (1,timestep,batch,depth)
        """
        self.inp = inp
        self.total_step = self.inp.shape[1]
        self.batch = self.inp.shape[2]
        self.pad_inp = self.pad_this_inp()
        self.reduce_total_step = int(self.total_step/self.reduce_size)
        output = np.zeros((1, self.reduce_total_step,
                           self.batch, self.output_depth)).astype(self.dtype)
        for this_step in range(self.reduce_total_step):
            for k_idx in range(self.reduce_size):
                weighted_pad_inp = np.dot(
                    self.pad_inp[0, this_step*self.reduce_size + k_idx], self.W[k_idx])
                output[0, this_step] += weighted_pad_inp
            output[0, this_step] += self.b
        return output

    @jit(fastmath=True)
    def backprop(self, dLoss):
        dL_prev = np.zeros_like(self.pad_inp)
        self.dW = np.zeros_like(self.W)

        self.db = np.zeros_like(self.b)
        normalization = self.total_step*self.batch
        self.db = np.einsum('tbd->d', dLoss[0])/normalization


        for idx in range(self.reduce_total_step):
            for k_idx in range(self.reduce_size):
                this_x = self.pad_inp[0, idx*self.reduce_size+k_idx]
                self.dW[k_idx] += np.dot(this_x.T, dLoss[0][idx])/normalization
                dL_prev[0, idx*self.reduce_size+k_idx] += np.dot(dLoss[0, idx],
                                                                 self.W[k_idx].T)


        dL_prev = dL_prev[:, self.amount_of_pad_front:self.total_step-self.amount_of_pad_end]
        return dL_prev

    def pad_this_inp(self):
        amount_of_pad = self.inp.shape[1] % self.reduce_size
        # if even, add equal amount on both sides, if odds, add more one pad at the front.
        self.amount_of_pad_front = amount_of_pad //2 + amount_of_pad%2
        self.amount_of_pad_end = amount_of_pad //2

        pads_front = np.tile(np.zeros((self.batch, self.input_depth)).astype(self.dtype),
                             (1, self.amount_of_pad_front, 1, 1))
        pads_end = np.tile(np.zeros((self.batch, self.input_depth)).astype(self.dtype),
                           (1, self.amount_of_pad_end, 1, 1))
        return np.concatenate((pads_front, self.inp, pads_end), axis=1)


class ExpandConv(trainable_layer):
    def __init__(self, hidden_units, paddling=True, dtype=np.float32):
        """
        hidden_units : in shape (expand_size, input_depth,output_depth)
        expand_size  :  input_timestep -> expand_size*input_timestep
        padding      : add zero paddlings at both sides of input.

        """
        self.dtype = dtype
        self.expand_size = hidden_units[0]
        self.input_depth = hidden_units[1]
        self.output_depth = hidden_units[2]
        self.W = (np.random.random(hidden_units)-0.5)/self.output_depth
        self.b = (np.random.random((self.output_depth))-0.5)/self.output_depth




        self.W = self.W.astype(self.dtype)
        self.b = self.b.astype(self.dtype)
        self.paddling = paddling

    @jit(fastmath=True)
    def forward(self, inp):
        """
        inp : input data in shape (1,timestep,batch,depth)
        """
        self.inp = inp
        self.total_step = self.inp.shape[1]
        self.batch = self.inp.shape[2]
        self.expand_total_step = int(self.total_step*self.expand_size)
        output = np.zeros((1, self.expand_total_step,
                           self.batch, self.output_depth)).astype(self.dtype)
        for this_step in range(self.total_step):
            for k_idx in range(self.expand_size):
                weighted_pad_inp = np.dot(self.inp[0, this_step], self.W[k_idx])
                output[0, this_step*self.expand_size+k_idx] = weighted_pad_inp + self.b
        return output

    @jit(fastmath=True)
    def backprop(self, dLoss):
        dL_prev = np.zeros_like(self.inp)
        self.dW = np.zeros_like(self.W)
        normalization = self.total_step*self.batch
        self.db = np.einsum('tbo->o', dLoss[0])/normalization

        for idx in range(self.total_step):
            for k_idx in range(self.expand_size):
                this_x = self.inp[0, idx]
                self.dW[k_idx] += np.dot(
                    this_x.T, dLoss[0, idx*self.expand_size+k_idx])/normalization
                dL_prev[0, idx] += np.dot(
                    dLoss[0, idx*self.expand_size+k_idx], self.W[k_idx].T)


        return dL_prev

class conv1D_rev(trainable_layer):
    """
    reverse version of conv1D
    """
    def __init__(self, hidden_units, paddling=True, stride=1, residual=False, dtype=np.float32):
        """
        hidden_units : in shape [input_filter,output_filter,kernel_size,input_depth,output_depth]
        padding      : add zero paddlings at both sides of input.
        stride       : displacement of one step.
        residual     : output = output + input  (short connection)
        """
        self.dtype = dtype
        self.input_filters = hidden_units[0]
        self.output_filters = hidden_units[1]
        self.kernel_size = hidden_units[2]
        self.input_depth = hidden_units[3]
        self.output_depth = hidden_units[4]
        normalization = self.input_filters*self.output_filters*self.kernel_size*self.input_depth
        self.W = (np.random.random(hidden_units)-0.5)/np.sqrt(normalization)
        self.b = (np.random.random((self.input_filters,
                                    self.output_filters,
                                    self.output_depth))-0.5)/np.sqrt(normalization)


        self.W = self.W.astype(self.dtype)
        self.b = self.b.astype(self.dtype)
        self.paddling = paddling
        self.stride = stride
        self.residual = residual

    @jit(fastmath=True)
    def forward(self, inp):
        self.inp = inp
        self.total_step = self.inp.shape[1]
        self.batch = self.inp.shape[2]
        amount_of_pad = self.kernel_size - 1
        self.amount_of_pad_front = amount_of_pad //2 + amount_of_pad%2
        self.amount_of_pad_end = amount_of_pad //2
        self.pad_total_step = amount_of_pad + self.total_step
        output = np.zeros((self.output_filters,
                           self.pad_total_step, self.batch, self.output_depth)).astype(self.dtype)
        for this_input_filter in range(self.input_filters):
            for this_output_filter in range(self.output_filters):
                for this_step in range(self.total_step):
                    for k_idx in range(self.kernel_size):
                        weighted_inp = np.dot(self.inp[this_input_filter, this_step],
                                              self.W[this_input_filter, this_output_filter, k_idx])
                        output[this_output_filter, this_step+k_idx] += \
                            weighted_inp+self.b[this_input_filter, this_output_filter]
        output = output[:, self.amount_of_pad_front:self.pad_total_step -self.amount_of_pad_end]

        if self.residual:
            output = output + self.inp
        return output
    @jit(fastmath=True)
    def backprop(self, dLoss):
        self.pad_dLoss = self.pad_this_dLoss(dLoss)
        dL_prev = np.zeros_like(self.inp)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.pad_dLoss_timestep = self.pad_dLoss.shape[1]
        normalization = self.total_step*self.batch*self.input_filters
        for this_input_filter in range(self.input_filters):
            for this_output_filter in range(self.output_filters):
                self.db[this_input_filter, this_output_filter] = np.einsum(
                    'tbo->o', dLoss[this_output_filter])/normalization
                for idx in range(self.kernel_size):
                    for this_inp, this_out in enumerate(range(
                            idx, self.pad_dLoss_timestep-(self.kernel_size-(idx+1)))):
                        this_x = self.pad_dLoss[this_output_filter, this_out]
                        self.dW[this_input_filter, this_output_filter, idx] += np.dot(
                            this_x.T, self.inp[this_input_filter, this_inp]).T/normalization

                for idx in range(self.total_step):
                    for k_idx in range(self.kernel_size):
                        this_dLoss = self.pad_dLoss[this_output_filter, idx+k_idx]
                        dL_prev[this_input_filter, idx] += np.dot(
                            this_dLoss, self.W[this_input_filter, this_output_filter, k_idx].T)
        if self.residual:
            dL_prev = dL_prev + dLoss
        return dL_prev
    def pad_this_dLoss(self, dLoss):
        pads_front = np.tile(np.zeros((self.batch, self.output_depth)).astype(self.dtype),
                             (self.output_filters, self.amount_of_pad_front, 1, 1))
        pads_end = np.tile(np.zeros((self.batch, self.output_depth)).astype(self.dtype),
                           (self.output_filters, self.amount_of_pad_end, 1, 1))
        return  np.concatenate((pads_front, dLoss, pads_end), axis=1)

class GLU:
    """
    gated  linear  units   ( arXiv:1612.08083v3  8 Sep 2017)
    Title : Language Modeling with Gated Convolutional Networks
    Author: Yann N. Dauphin, Angela Fan, Michael Auli, David Grangier
    inp [AB] = A * sigmoid(B)
    """
    def __init__(self):
        self.Sig = Sigmoid()

    def forward(self, inp):
        """
        input data in shape (amount of filters, timestep, batch, *depth)
        *depth : input part from depth*0.0 to depth*0.5
                 gate  part from depth*0.5 to depth*1.0
        """
        half_depth = int(inp.shape[3]/2)
        self.info_A = inp[:, :, :, :half_depth]
        self.gate_B = self.Sig.forward(inp[:, :, :, half_depth:])
        output = self.info_A * self.gate_B
        return  output
    def backprop(self, dLoss):
        dinfo_A = dLoss*self.gate_B
        dgate_B = self.info_A*self.Sig.backprop(dLoss)
        dL_prev = np.concatenate((dinfo_A, dgate_B), axis=3)
        return dL_prev
    def description(self):
        return " A*sigmoid(B) (2*depth -> depth)"

class GTU:
    """
    inp [AB] = tanh(A) * sigmoid(B)
    """
    def __init__(self):
        self.Sig = Sigmoid()
        self.Tanh = Tanh()

    def forward(self, inp):
        """
        input data in shape (amount of filters, timestep, batch, *depth)
        *depth : input part from depth*0.0 to depth*0.5
                 gate  part from depth*0.5 to depth*1.0
        """
        half_depth = int(inp.shape[3]/2)
        self.info_A = self.Tanh.forward(inp[:, :, :, :half_depth])
        self.gate_B = self.Sig.forward(inp[:, :, :, half_depth:])
        output = self.info_A * self.gate_B
        return output
    def backprop(self, dLoss):
        dinfo_A = self.gate_B*self.Tanh.backprop(dLoss)
        dgate_B = self.info_A*self.Sig.backprop(dLoss)
        dL_prev = np.concatenate((dinfo_A, dgate_B), axis=3)
        return dL_prev
    def description(self):
        return " tanh(A)*sigmoid(B) (2*depth -> depth)"

class flatten:
    def forward(self, inp):
        self.original_shape = inp.shape
        self.timestep = inp.shape[1]
        self.batch = inp.shape[2]
        self.depth = inp.shape[3]
        return inp.transpose(0, 2, 1, 3).reshape(self.batch, -1)
    def backprop(self, dLoss):
        return dLoss.reshape(1, self.batch, self.timestep, self.depth).transpose(0, 2, 1, 3)
    def description(self):
        return "(T,B,D)->(B,T*D)"

class rever_flatten:
    def __init__(self, timestep, depth):
        self.timestep = timestep
        self.depth = depth
    def forward(self, inp):
        self.batch = inp.shape[0]
        return inp.reshape(1, self.batch, self.timestep, self.depth).transpose(0, 2, 1, 3)
    def backprop(self, dLoss):
        return dLoss.transpose(0, 2, 1, 3).reshape(self.batch, -1)
    def description(self):
        return "(B,T*D)->(T,B,D)"


class expand_dims:
    def __init__(self, expand_at):
        self.expand_at = expand_at
    def forward(self, inp):
        return  np.expand_dims(inp, self.expand_at)
    def backprop(self, dLoss):
        return  np.squeeze(dLoss, self.expand_at)
    def description(self):
        return "(T,B,D)->(1,T,B,D)"
class squeeze:
    def __init__(self, squeeze_at):
        self.squeeze_at = squeeze_at
    def forward(self, inp):
        return  np.squeeze(inp, self.squeeze_at)
    def backprop(self, dLoss):
        return  np.expand_dims(dLoss, self.squeeze_at)
    def description(self):
        return "(1,T,B,D)->(T,B,D)"

class split:
    def __init__(self, cutoff_at):
        self.cutoff_at = cutoff_at
    def forward(self, inp):
        self.front_part = inp[:, :, :, :self.cutoff_at]
        self.end_part = inp[:, :, :, self.cutoff_at:]
        self.empty_pad = np.zeros_like(inp[:, :, :, self.cutoff_at:])
        return  self.front_part
    def get_both(self):
        return self.front_part, self.end_part
    def backprop(self, dLoss):
        return dLoss
        #return  np.concatenate((dLoss,self.empty_pad),axis=3)
    def description(self):
        return "(F,T,B,D)->(F,T,B,:slice)"

class concat:
    def forward(self, inp1, inp2):
        self.depth_of_inp1 = inp1.shape[3]
        return  np.concatenate((inp1, inp2), axis=3)
    def backprop(self, dLoss):
        return  dLoss[:, :, :, :self.depth_of_inp1], dLoss[:, :, :, self.depth_of_inp1:]
    def description(self):
        return "(F,T,B,d1) (F,T,B,d2)->(F,T,B,d1+d2)"

