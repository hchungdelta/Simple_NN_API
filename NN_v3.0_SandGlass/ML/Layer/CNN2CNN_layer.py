import time
import numpy as np
from numba import njit, jit
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
    def __init__(self, hidden_units, paddling=True, stride=1, residual=False):
        """
        hidden_units : in shape [input_filter,output_filter,kernel_size,input_depth,output_depth]
        padding      : add zero paddlings at both sides of input.
        stride       : displacement of one step.
        residual     : output = output + input  (short connection)
        """
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
        output = np.zeros((self.output_filters, self.total_step, self.batch, self.output_depth))
        'O:output filters , k: kernel size, b: batch, d: input_depth, o: output_depth'
        for this_input_filter in range(self.input_filters):
            for this_output_filter in range(self.output_filters):
                for this_step in range(self.total_step):
                    weighted_pad_inp = np.einsum('kbi,kio->bo',
                                                 self.pad_inp[this_input_filter,
                                                              this_step:self.kernel_size+this_step],
                                                 self.W[this_input_filter, this_output_filter])
                    output[this_output_filter, this_step] += weighted_pad_inp +\
                        self.b[this_input_filter, this_output_filter]
        if self.residual:
            output = output + self.inp
        return output
    @jit(fastmath=True)
    def backprop(self, dLoss):
        dL_prev = np.zeros_like(self.pad_inp)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.pad_inp_timestep = self.pad_inp.shape[1]
        'b for batch, d for input_depth, o for output_depth , t for timestep'
        normalization = self.total_step*self.batch*self.output_filters
        for this_input_filter in range(self.input_filters):
            for this_output_filter in range(self.output_filters):
                self.db[this_input_filter, this_output_filter] = np.einsum(
                    'tbd->d', dLoss[this_output_filter])/normalization

                for idx in range(self.kernel_size):
                    this_x = self.pad_inp[this_input_filter,
                                          idx:self.pad_inp_timestep-(self.kernel_size-(idx+1))]
                    self.dW[this_input_filter, this_output_filter, idx] += np.einsum(
                        'tbd,tbo->do', this_x, dLoss[this_output_filter])/normalization

                ' dLoss of output timestep -> amount of kernel size of (paddling) input timestep'
                for idx in range(self.total_step):
                    dL_prev[this_input_filter, idx:idx+self.kernel_size] += np.einsum(
                        'bo,kdo->kbd', dLoss[this_output_filter, idx],
                        self.W[this_input_filter, this_output_filter])
        #self.dW = np.einsum('ktbo,tbd->kod',this_x,dLoss)/(self.total_step*self.batch)
        # slice the paddling parts.
        dL_prev = dL_prev[:, self.amount_of_pad_front:self.pad_total_step-self.amount_of_pad_end]
        if self.residual:
            dL_prev = dL_prev + dLoss
        return dL_prev


    def pad_this_inp(self):
        amount_of_pad = self.kernel_size - 1
        # if even, add equal amount on both sides, if odds, add more one pad at the front.
        self.amount_of_pad_front = amount_of_pad //2  + amount_of_pad % 2
        self.amount_of_pad_end = amount_of_pad //2

        pads_front = np.tile(np.zeros((self.batch, self.input_depth)),
                             (self.input_filters, self.amount_of_pad_front, 1, 1))
        pads_end = np.tile(np.zeros((self.batch, self.input_depth)),
                           (self.input_filters, self.amount_of_pad_end, 1, 1))
        return  np.concatenate((pads_front, self.inp, pads_end), axis=1)


class conv1D_group():
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
    def forward(self, inp):
        self.inp = inp
        self.inp_conv1D_amount = inp.shape[0]
        sum_output = []
        for idx, this_conv1D in enumerate(self.conv1Ds):
            sum_output.append(this_conv1D.forward(np.array([inp[idx]])))
        return  np.sum(sum_output,axis=0)

    def backprop(self, dLoss):
        dL_prev = np.zeros_like(self.inp)
        for idx, this_conv1D in enumerate(self.conv1Ds):
            dL_prev[idx] = this_conv1D.backprop(dLoss)
        return  dL_prev





class ReduceAttn():
    def __init__(self, reduce_size, paddling=True):
        """
        reduce_size  : input_timestep -> input_timestep/reduce_size
        padding      : add zero paddlings at both sides of input.

        """
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
        self.alpha = np.zeros((self.reduce_total_step, self.batch, self.reduce_size))
        output = np.zeros((1, self.reduce_total_step, self.batch, self.input_depth))
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

        pads_front = np.tile(np.zeros((self.batch, self.input_depth)),
                             (1, self.amount_of_pad_front, 1, 1))
        pads_end = np.tile(np.zeros((self.batch, self.input_depth)),
                           (1, self.amount_of_pad_end, 1, 1))
        return  np.concatenate((pads_front, self.inp, pads_end), axis=1)
    def give_me_alpha(self):
        return self.alpha



class ReduceConv(trainable_layer):
    def __init__(self, hidden_units, paddling=True):
        """
        hidden_units : in shape (reduce_size, input_depth,output_depth)
        reduce_size  : input_timestep -> input_timestep/reduce_size
        padding      : add zero paddlings at both sides of input.
        
        """
        self.reduce_size = hidden_units[0]
        self.input_depth = hidden_units[1]
        self.output_depth = hidden_units[2]
        self.W = (np.random.random(hidden_units)-0.5) /np.sqrt(self.reduce_size)
        self.b = (np.random.random((self.output_depth))-0.5)/np.sqrt(self.reduce_size)
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

        output = np.zeros((1, self.reduce_total_step, self.batch, self.output_depth))

        'k: kernel size, b: batch, d: input_depth, o: output_depth'
        for this_step in range(self.reduce_total_step):
            weighted_pad_inp = np.einsum(
                'kbd,kdo->bo',
                self.pad_inp[0, this_step*self.reduce_size:(1+this_step)*self.reduce_size],
                self.W)
            output[0, this_step] = weighted_pad_inp + self.b
        return output
    @jit(fastmath=True)
    def backprop(self, dLoss):
        dL_prev = np.zeros_like(self.pad_inp)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        'b for batch, d for input_depth, o for output_depth , t for timestep'
        normalization = self.total_step*self.batch
        self.db = np.einsum('tbd->d', dLoss[0])/normalization

        for idx in range(self.reduce_total_step):
            this_x = self.pad_inp[0, idx*self.reduce_size:(idx+1)*self.reduce_size]
            self.dW += np.einsum('tbd,bo->tdo', this_x, dLoss[0][idx])/normalization
            dL_prev[0, idx*self.reduce_size:(idx+1)*self.reduce_size] = np.einsum('bo,kdo->kbd',
                                                                                  dLoss[0, idx],
                                                                                  self.W)
        dL_prev = dL_prev[:, self.amount_of_pad_front:self.total_step-self.amount_of_pad_end]
        return dL_prev
    def pad_this_inp(self):
        amount_of_pad = self.inp.shape[1] % self.reduce_size
        # if even, add equal amount on both sides, if odds, add more one pad at the front.
        self.amount_of_pad_front = amount_of_pad //2 + amount_of_pad%2
        self.amount_of_pad_end = amount_of_pad //2

        pads_front = np.tile(np.zeros((self.batch, self.input_depth)),
                             (1, self.amount_of_pad_front, 1, 1))
        pads_end = np.tile(np.zeros((self.batch, self.input_depth)),
                           (1, self.amount_of_pad_end, 1, 1))
        return np.concatenate((pads_front, self.inp, pads_end), axis=1)


class ExpandConv(trainable_layer):
    def __init__(self, hidden_units, paddling=True):
        """
        hidden_units : in shape (expand_size, input_depth,output_depth)
        expand_size  :  input_timestep -> expand_size*input_timestep
        padding      : add zero paddlings at both sides of input.

        """
        self.expand_size = hidden_units[0]
        self.input_depth = hidden_units[1]
        self.output_depth = hidden_units[2]
        self.W = (np.random.random(hidden_units)-0.5)
        self.b = (np.random.random((self.output_depth))-0.5)
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
        output = np.zeros((1, self.expand_total_step, self.batch, self.output_depth))

        'k: kernel size, b: batch, d: input_depth, o: output_depth'
        for this_step in range(self.total_step):
            weighted_pad_inp = np.einsum('bd,kdo->kbo', self.inp[0, this_step], self.W)
            output[0, this_step*self.expand_size:(1+this_step)*self.expand_size] = weighted_pad_inp + self.b
        return output
    @jit(fastmath=True)
    def backprop(self, dLoss):
        dL_prev = np.zeros_like(self.inp)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        'b for batch, d for input_depth, o for output_depth , t for timestep'
        normalization = self.total_step*self.batch
        self.db = np.einsum('tbo->o', dLoss[0])/normalization

        for idx in range(self.total_step):
            this_x = self.inp[0, idx]
            self.dW += np.einsum(
                'bd,kbo->kdo', this_x, dLoss[0, idx*self.expand_size:(idx+1)*self.expand_size])/normalization
            dL_prev[0, idx] += np.einsum(
                'kbo,kdo->bd', dLoss[0, idx*self.expand_size:(idx+1)*self.expand_size], self.W)
        return dL_prev

class conv1D_rev(trainable_layer):
    """
    reverse version of conv1D
    """
    def __init__(self, hidden_units, paddling=True, stride=1, residual=False):
        """
        hidden_units : in shape [input_filter,output_filter,kernel_size,input_depth,output_depth]
        padding      : add zero paddlings at both sides of input.
        stride       : displacement of one step.
        residual     : output = output + input  (short connection)
        """
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
        output = np.zeros((self.output_filters, self.pad_total_step, self.batch, self.output_depth))

        'k: kernel size, b: batch, d: input_depth, o: output_depth'
        for this_input_filter in range(self.input_filters):
            for this_output_filter in range(self.output_filters):
                for this_step in range(self.total_step):
                    weighted_inp = np.einsum('bi,kio->kbo',
                                             self.inp[this_input_filter, this_step],
                                             self.W[this_input_filter, this_output_filter])
                    output[this_output_filter, this_step:this_step+self.kernel_size] += \
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
        'b for batch, d for input_depth, o for output_depth , t for timestep'
        normalization = self.total_step*self.batch*self.input_filters
        for this_input_filter in range(self.input_filters):
            for this_output_filter in range(self.output_filters):
                self.db[this_input_filter, this_output_filter] = np.einsum(
                    'tbo->o', dLoss[this_output_filter])/normalization
                for idx in range(self.kernel_size):
                    this_x = self.pad_dLoss[this_output_filter,
                                            idx:self.pad_dLoss_timestep-(self.kernel_size-(idx+1))]
                    self.dW[this_input_filter, this_output_filter, idx] += np.einsum(
                        'tbo,tbi->io', this_x, self.inp[this_input_filter])/normalization
                for idx in range(self.total_step):
                    dL_prev[this_input_filter, idx] += np.einsum(
                        'kbo,kdo->bd', self.pad_dLoss[this_output_filter, idx:idx+self.kernel_size],
                        self.W[this_input_filter, this_output_filter])
        if self.residual:
            dL_prev = dL_prev + dLoss
        return dL_prev
    def pad_this_dLoss(self, dLoss):
        pads_front = np.tile(np.zeros((self.batch, self.output_depth)),
                             (self.output_filters, self.amount_of_pad_front, 1, 1))
        pads_end = np.tile(np.zeros((self.batch, self.output_depth)),
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
        return   output      
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
class BatchNorm(trainable_layer):
    def __init__(self,mode='train'):
        self.W = 1.0-np.random.random(1)*0.1  # rquals to gamma
        self.b = np.random.random(1)*0.1  # equals to beta
        self.mode = mode
        self.recent_var = []
        self.recent_mean = []
        self.record_amount = 10
        if self.mode == "infer":
            pass 
    def forward(self, inp):
        self.inp = inp
        self.inp_filters = inp.shape[0]
        self.timestep = inp.shape[1]
        self.batch = inp.shape[2]
        self.depth = inp.shape[3]
        self.normalization = 1./(self.batch*self.timestep*self.inp_filters)

        if self.mode == 'train':
            # mini-batch mean
            self.mb_mean = np.einsum('ftbd->d', self.inp)*self.normalization
            self.diff = self.inp - self.mb_mean
            self.mb_var = np.einsum('ftbd->d', self.diff**2)*self.normalization
            self.eps = np.ones_like(self.mb_mean)*1e-8
            self.sqrt_var = np.sqrt(self.mb_var + self.eps)
            self.norm_x = self.diff/self.sqrt_var
            output = self.W*self.norm_x +self.b
            self.recent_mean.insert(0,self.mb_mean)
            self.recent_var.insert(0,self.sqrt_var)
            if len(self.recent_mean) > self.record_amount:
                self.recent_mean = self.recent_mean[:self.record_amount]
                self.recent_var = self.recent_var[:self.record_amount]
        if self.mode == 'infer':
            self.diff = self.inp - self.infer_mean
            self.norm_x = self.diff/self.infer_var
            output = self.W*self.norm_x +self.b
        return output
    def backprop(self, dLoss):
        self.db = np.sum(dLoss)*self.normalization*(1./self.depth)
        self.dW = np.sum(dLoss*self.norm_x)*self.normalization*(1./self.depth)
        d_norm_x = dLoss*self.W
        d_mb_var = -0.5*d_norm_x*self.diff*(self.sqrt_var**(-3))
        d_mb_mean = (-1*d_norm_x/self.sqrt_var)+d_mb_var*(-2)*self.diff*self.normalization
        dL_prev = (d_norm_x/self.sqrt_var) + (d_mb_var*self.diff*2+ d_mb_mean)*self.normalization
        return dL_prev
     
    def rewrite_parameter(self, receive_parameter):
        self.infer_mean = receive_parameter["infer_mean"]
        self.infer_var = receive_parameter["infer_var"]

    def get_parameter(self):
        # infer var is already in form np.sqrt(var+eps)
        infer_mean = np.mean(np.array(self.recent_mean),axis=0)
        infer_var = np.mean(np.array(self.recent_var),axis=0)
        parameter = {"infer_mean":infer_mean,"infer_var":infer_var}
        return parameter
     


class BatchNorm_FCL(trainable_layer):
    def __init__(self):
        self.W = 1.0-np.random.random(1)*0.1  # equals to gamma
        self.b = np.random.random(1)*0.1  # equals to beta

    def forward(self, inp):
        self.inp = inp
        self.batch = inp.shape[0]
        self.depth = inp.shape[1]
        self.normalization = (1./self.batch)
        # mini-batch mean
        self.mb_mean = np.einsum('bd->d', self.inp)*self.normalization
        self.diff = self.inp - self.mb_mean
        self.mb_var = np.einsum('bd->d', self.diff**2)*self.normalization
        self.eps = np.ones_like(self.mb_mean)*1e-5
        self.sqrt_var = np.sqrt(self.mb_var + self.eps)
        self.norm_x = self.diff/self.sqrt_var
        output = self.W*self.norm_x + self.b
        return output
    def backprop(self, dLoss):
        self.db = np.sum(dLoss)*self.normalization#*(1./self.depth)
        self.dW = np.sum(dLoss*self.norm_x)*self.normalization#*(1./self.depth)
        d_norm_x = dLoss*self.W
        d_mb_var = -0.5*d_norm_x*self.diff*(self.sqrt_var**(-3))
        d_mb_mean = (-1*d_norm_x/self.sqrt_var)+d_mb_var*(-2)*self.diff*self.normalization
        dL_prev = (d_norm_x/self.sqrt_var) + (d_mb_var*self.diff*2+ d_mb_mean)*self.normalization
        return dL_prev

class flatten:
    def forward(self, inp):
        self.original_shape = inp.shape
        self.batch = inp.shape[2]
        return inp.reshape(self.batch, -1)
    def backprop(self, dLoss):
        return dLoss.reshape(self.original_shape)
    def description(self):
        return "(T,B,D)->(B,T*D)"
 
class rever_flatten:
    def __init__(self, into_shape):
        self.into_shape = into_shape
    def forward(self, inp):
        self.original_shape = inp.shape
        return inp.reshape(self.into_shape)
    def backprop(self, dLoss):
        return dLoss.reshape(self.original_shape)
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

