"""
Batch Normalization (https://arxiv.org/abs/1502.03167)
To train the model more efficiently.
"""
import numpy as np

class trainable_layer:
    """
    set as trainable variables.
    """
    def __init__(self):
        self.dW = None
        self.db = None
        self.W = None
        self.b = None
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

class LSTM_BatchNorm(trainable_layer):
    """
    input in shape ( j, k, z)
    the mean/var is calculated by averging j*k
    """
    def __init__(self, hidden_units, eps=1e-6, mode='train', dtype=np.float32):
        """
        create 2 vectors, weight and bias in shape hidden_units
        mode = train or infer
        """
        self.W = 1-np.random.random((hidden_units))*0.2  # equals to gamma
        self.b = np.random.random((hidden_units))*0.2  # equals to beta
        self.W = self.W.astype(dtype)
        self.b = self.b.astype(dtype)
        self.mode = mode
        self.eps = eps
        self.recent_var = []
        self.recent_mean = []
        self.record_amount = 12
        self.timestep_mb_mean = []
        self.timestep_diff = []
        self.timestep_sqrt_var = []
        self.timestep_norm_x = []
        self.mb_mean = None
        self.diff = None
        self.sqrt_var = None
        self.norm_x = None

        self.use_timestep_mode = False
        self.timestep_backprop_init = False
        self.counter = 0
    def forward(self, inp):
        self.timestep = inp.shape[0]
        self.batch = inp.shape[1]
        self.normalization = 1./(self.batch)

        if self.mode == 'train':
            self.mb_mean = np.einsum('tbd->td', inp)*self.normalization
            tiled_mb_mean = np.tile(self.mb_mean, (self.batch, 1, 1)).transpose(1, 0, 2)
            self.diff = inp - tiled_mb_mean
            self.mb_var = np.einsum('tbd->td', self.diff**2)*self.normalization
            self.sqrt_var = np.sqrt(self.mb_var + self.eps)
            self.tiled_sqrt_var = np.tile(self.sqrt_var, (self.batch, 1, 1)).transpose(1, 0, 2)
            self.norm_x = self.diff/self.tiled_sqrt_var
            output = self.W*self.norm_x +self.b
            self.recent_mean.insert(0, self.mb_mean)
            self.recent_var.insert(0, self.sqrt_var)
            if len(self.recent_mean) > self.record_amount:
                self.recent_mean = self.recent_mean[:self.record_amount]
                self.recent_var = self.recent_var[:self.record_amount]

        if self.mode == 'infer':
            _infer_mean = self.infer_mean[:inp.shape[0]]
            _infer_var = self.infer_var[:inp.shape[0]]
            self.diff = inp - np.tile(_infer_mean, (self.batch, 1, 1)).transpose(1, 0, 2)
            self.norm_x = self.diff/np.tile(_infer_var, (self.batch, 1, 1)).transpose(1, 0, 2)
            output = self.W*self.norm_x +self.b
        return output

    def timestep_forward(self, inp):
        self.use_timestep_mode = True
        self.timestep_backprop_init = False
        self.batch = inp.shape[0]
        self.normalization = 1./(self.batch)

        if self.mode == 'train':
            this_t_mb_mean = np.einsum('bd->d', inp)*self.normalization
            this_t_diff = inp - this_t_mb_mean
            this_t_mb_var = np.einsum('bd->d', this_t_diff**2)*self.normalization
            this_t_sqrt_var = np.sqrt(this_t_mb_var + self.eps)
            this_t_norm_x = this_t_diff/this_t_sqrt_var

            output = self.W*this_t_norm_x +self.b
            self.timestep_mb_mean.append(this_t_mb_mean)
            self.timestep_diff.append(this_t_diff)
            self.timestep_sqrt_var.append(this_t_sqrt_var)
            self.timestep_norm_x.append(this_t_norm_x)
            self.counter += 1

        if self.mode == 'infer':
            _infer_mean = self.infer_mean[self.counter]
            _infer_var = self.infer_var[self.counter]
            self.diff = inp - np.tile(_infer_mean, (self.batch, 1))
            self.norm_x = self.diff/np.tile(_infer_var, (self.batch, 1))
            output = self.W*self.norm_x +self.b
            self.counter += 1
        return output

    def backprop(self, dLoss):
        if self.use_timestep_mode == True:
            self.timestep_gather()
        self.db = np.einsum('tbd->d', dLoss)
        self.dW = np.einsum('tbd->d', dLoss*self.norm_x)
        std_inv = 1./self.sqrt_var
        d_norm_x = dLoss*self.W
        d_mb_var = -0.5*np.einsum('tbd->td', d_norm_x*self.diff)*std_inv**3
        d_mb_mean = np.einsum('tbd,td->td', -1*d_norm_x, std_inv) + \
            d_mb_var*np.einsum('tbd->td', -2*self.diff)*self.normalization
        dL_prev = np.einsum('tbd,td->tbd', d_norm_x, std_inv) +\
                  np.einsum('td,tbd->tbd', d_mb_var, self.diff*2)*self.normalization +\
                  np.tile(d_mb_mean, (self.batch, 1, 1)).transpose(1, 0, 2)*self.normalization
        return dL_prev

    def timestep_backprop(self, dLoss):
        if not self.timestep_backprop_init:
            self.timestep_gather()
            self.timestep_backprop_init = True
            self.db = np.einsum('bd->d', dLoss)
            self.dW = np.einsum('bd->d', dLoss*self.norm_x[self.counter])
        if self.timestep_backprop_init:
            self.db += np.einsum('bd->d', dLoss)
            self.dW += np.einsum('bd->d', dLoss*self.norm_x[self.counter])

        std_inv = 1./self.sqrt_var[self.counter]
        d_norm_x = dLoss*self.W
        d_mb_var = -0.5*np.einsum('bd->d', d_norm_x*self.diff[self.counter])*std_inv**3
        d_mb_mean = np.einsum('bd,d->d', -1*d_norm_x, std_inv) + \
            d_mb_var*np.einsum('bd->d', -2*self.diff[self.counter])*self.normalization
        dL_prev = np.einsum('bd,d->bd', d_norm_x, std_inv) +\
                  np.einsum('d,bd->bd', d_mb_var, self.diff[self.counter]*2)*self.normalization +\
                  np.tile(d_mb_mean, (self.batch, 1))*self.normalization
        return dL_prev


    def timestep_gather(self):
        self.mb_mean = np.array(self.timestep_mb_mean)
        self.diff = np.array(self.timestep_diff)
        self.sqrt_var = np.array(self.timestep_sqrt_var)
        self.norm_x = np.array(self.timestep_norm_x)
        self.timestep_mb_mean = []
        self.timestep_diff = []
        self.timestep_sqrt_var = []
        self.timestep_norm_x = []
        self.recent_mean.insert(0, self.mb_mean)
        self.recent_var.insert(0, self.sqrt_var)
        self.counter = 0
        if len(self.recent_mean) > self.record_amount:
            self.recent_mean = self.recent_mean[:self.record_amount]
            self.recent_var = self.recent_var[:self.record_amount]


    def rewrite_parameter(self, receive_parameter):
        self.infer_mean = receive_parameter["infer_mean"]
        self.infer_var = receive_parameter["infer_var"]

    def get_parameter(self):
        # infer var is already in form np.sqrt(var+eps)
        infer_mean = np.mean(np.array(self.recent_mean), axis=0)
        infer_var = np.mean(np.array(self.recent_var), axis=0)
        parameter = {"infer_mean":infer_mean, "infer_var":infer_var}
        return parameter

class BatchNorm_4d(trainable_layer):
    """
    input in shape (i, j, k, z)
    the mean/var is calculated by averging i*j*k
    """
    def __init__(self, hidden_units, eps, mode='train', dtype=np.float32):
        """
        create 2 vectors, weight and bias in shape hidden_units
        mode = train or infer
        """
        self.W = 1-np.random.random((hidden_units))*0.2  # equals to gamma
        self.b = np.random.random((hidden_units))*0.2  # equals to beta
        self.W = self.W.astype(dtype)
        self.b = self.b.astype(dtype)
        self.mode = mode
        self.eps = eps
        self.recent_var = []
        self.recent_mean = []
        self.record_amount = 10

    def forward(self, inp):
        self.inp_filters = inp.shape[0]
        self.timestep = inp.shape[1]
        self.batch = inp.shape[2]
        self.normalization = 1./(self.batch*self.timestep*self.inp_filters)

        if self.mode == 'train':
            self.mb_mean = np.einsum('ftbd->d', inp)*self.normalization
            self.diff = inp - self.mb_mean
            self.mb_var = np.einsum('ftbd->d', self.diff**2)*self.normalization

            self.sqrt_var = np.sqrt(self.mb_var + self.eps)
            self.norm_x = self.diff/self.sqrt_var
            output = self.W*self.norm_x +self.b
            self.recent_mean.insert(0, self.mb_mean)
            self.recent_var.insert(0, self.sqrt_var)
            if len(self.recent_mean) > self.record_amount:
                self.recent_mean = self.recent_mean[:self.record_amount]
                self.recent_var = self.recent_var[:self.record_amount]

        if self.mode == 'infer':
            self.diff = inp - self.infer_mean
            self.norm_x = self.diff/self.infer_var
            output = self.W*self.norm_x +self.b
        return output

    def backprop(self, dLoss):
        self.db = np.einsum('ftbd->d', dLoss)
        self.dW = np.einsum('ftbd->d', dLoss*self.norm_x)
        std_inv = 1./self.sqrt_var
        d_norm_x = dLoss*self.W
        d_mb_var = -0.5*np.einsum('ftbd->d', d_norm_x*self.diff)*(std_inv**3)
        d_mb_mean = np.einsum('ftbd->d', -1*d_norm_x*std_inv) + \
            d_mb_var*np.einsum('ftbd->d', -2*self.diff)*self.normalization
        dL_prev = d_norm_x*std_inv +\
            (d_mb_var*self.diff*2+ d_mb_mean)*self.normalization
        return dL_prev

    def rewrite_parameter(self, receive_parameter):
        self.infer_mean = receive_parameter["infer_mean"]
        self.infer_var = receive_parameter["infer_var"]

    def get_parameter(self):
        # infer var is already in form np.sqrt(var+eps)
        infer_mean = np.mean(np.array(self.recent_mean), axis=0)
        infer_var = np.mean(np.array(self.recent_var), axis=0)
        parameter = {"infer_mean":infer_mean, "infer_var":infer_var}
        return parameter

class BatchNorm_3d(trainable_layer):
    """
    input in shape ( j, k, z)
    the mean/var is calculated by averging j*k
    """
    def __init__(self, hidden_units, eps=1e-6, mode='train', dtype=np.float32):
        """
        create 2 vectors, weight and bias in shape hidden_units
        mode = train or infer
        """
        self.W = 1-np.random.random((hidden_units))*0.2  # equals to gamma
        self.b = np.random.random((hidden_units))*0.2  # equals to beta
        self.W = self.W.astype(dtype)
        self.b = self.b.astype(dtype)
        self.mode = mode
        self.eps = eps
        self.recent_var = []
        self.recent_mean = []
        self.record_amount = 10

    def forward(self, inp):
        self.timestep = inp.shape[0]
        self.batch = inp.shape[1]
        self.normalization = 1./(self.batch*self.timestep)
        if self.mode == 'train':
            self.mb_mean = np.einsum('tbd->d', inp)*self.normalization
            self.diff = inp - self.mb_mean
            self.mb_var = np.einsum('tbd->d', self.diff**2)*self.normalization

            self.sqrt_var = np.sqrt(self.mb_var + self.eps)
            self.norm_x = self.diff/self.sqrt_var
            output = self.W*self.norm_x +self.b
            self.recent_mean.insert(0, self.mb_mean)
            self.recent_var.insert(0, self.sqrt_var)
            if len(self.recent_mean) > self.record_amount:
                self.recent_mean = self.recent_mean[:self.record_amount]
                self.recent_var = self.recent_var[:self.record_amount]

        if self.mode == 'infer':
            self.diff = inp - self.infer_mean
            self.norm_x = self.diff/self.infer_var
            output = self.W*self.norm_x +self.b

        return output
    def backprop(self, dLoss):
        self.db = np.einsum('tbd->d', dLoss)
        self.dW = np.einsum('tbd->d', dLoss*self.norm_x)
        std_inv = 1./self.sqrt_var
        d_norm_x = dLoss*self.W
        d_mb_var = -0.5*np.einsum('tbd->d', d_norm_x*self.diff)*(std_inv**3)
        d_mb_mean = np.einsum('tbd->d', -1*d_norm_x*std_inv) + \
            d_mb_var*np.einsum('tbd->d', -2*self.diff)*self.normalization
        dL_prev = d_norm_x*std_inv +\
            (d_mb_var*self.diff*2+ d_mb_mean)*self.normalization
        return dL_prev

    def rewrite_parameter(self, receive_parameter):
        self.infer_mean = receive_parameter["infer_mean"]
        self.infer_var = receive_parameter["infer_var"]

    def get_parameter(self):
        # infer var is already in form np.sqrt(var+eps)
        infer_mean = np.mean(np.array(self.recent_mean), axis=0)
        infer_var = np.mean(np.array(self.recent_var), axis=0)
        parameter = {"infer_mean":infer_mean, "infer_var":infer_var}
        return parameter

class BatchNorm_2d(trainable_layer):
    """
    input in shape (i, j)
    the mean/var is calculated by averging i
    """
    def __init__(self, hidden_units, eps=1e-5, mode='train', dtype=np.float32):
        """
        create 2 vectors, weight and bias in shape hidden_units
        mode = train or infer
        """
        self.W = 1-np.random.random((hidden_units))*0.2  # equals to gamma
        self.b = np.random.random((hidden_units))*0.2  # equals to beta
        self.W = self.W.astype(dtype)
        self.b = self.b.astype(dtype)
        self.mode = mode
        self.recent_var = []
        self.recent_mean = []
        self.record_amount = 10
        self.eps = eps
        self.diff = None
        self.norm_x = None
        self.sqrt_var = None
        self.normalization = None

    def forward(self, inp):
        self.batch = inp.shape[0]
        self.normalization = (1./self.batch)

        if self.mode == 'train':
            self.mb_mean = np.mean(inp, axis=0)
            self.diff = inp - self.mb_mean
            self.mb_var = np.var(inp, axis=0)
            self.sqrt_var = np.sqrt(self.mb_var + self.eps)
            self.norm_x = self.diff/self.sqrt_var
            output = self.W*self.norm_x + self.b
            self.recent_mean.insert(0, self.mb_mean)
            self.recent_var.insert(0, self.sqrt_var)
            if len(self.recent_mean) > self.record_amount:
                self.recent_mean = self.recent_mean[:self.record_amount]
                self.recent_var = self.recent_var[:self.record_amount]
        if self.mode == 'infer':
            self.diff = inp - self.infer_mean
            self.norm_x = self.diff/self.infer_var
            output = self.W*self.norm_x +self.b
        return output

    def backprop(self, dLoss):
        self.db = np.sum(dLoss, axis=0)
        self.dW = np.sum(dLoss*self.norm_x, axis=0)
        std_inv = 1./self.sqrt_var
        d_norm_x = dLoss*self.W
        d_mb_var = -0.5*np.sum(d_norm_x*self.diff, axis=0)*std_inv**3
        d_mb_mean = np.sum(-1*d_norm_x*std_inv, axis=0) + d_mb_var*np.mean(-2*self.diff, axis=0)
        dL_prev = d_norm_x*std_inv + (d_mb_var*self.diff*2+ d_mb_mean)*self.normalization
        return dL_prev

    def rewrite_parameter(self, receive_parameter):
        self.infer_mean = receive_parameter["infer_mean"]
        self.infer_var = receive_parameter["infer_var"]

    def get_parameter(self):
        # infer var is already in form np.sqrt(var+eps)
        infer_mean = np.mean(np.array(self.recent_mean), axis=0)
        infer_var = np.mean(np.array(self.recent_var), axis=0)
        parameter = {"infer_mean":infer_mean, "infer_var":infer_var}
        return parameter


class LSTM_LayerNorm(trainable_layer):
    def __init__(self, hidden_units, eps=1e-6, dtype=np.float32):
        self.W = 1-np.random.random((hidden_units))*0.2  # equals to gamma
        self.b = np.random.random((hidden_units))*0.2  # equals to beta
        self.W = self.W.astype(dtype)
        self.b = self.b.astype(dtype)
        self.depth = hidden_units
        self.eps = eps
        self.timestep_mb_mean = []
        self.timestep_diff = []
        self.timestep_sqrt_var = []
        self.timestep_norm_x = []
        self.mb_mean = None
        self.diff = None
        self.sqrt_var = None
        self.norm_x = None
        self.use_timestep_mode = False
        self.timestep_backprop_init = False
        self.counter = 0
        self.counter_backprop = 0
    def forward(self, inp):
        self.timestep = inp.shape[0]
        self.batch = inp.shape[1]
        self.normalization = 1./self.depth

        self.mb_mean = np.einsum('tbd->tb', inp)*self.normalization
        tiled_mb_mean = np.tile(self.mb_mean, (self.depth, 1, 1)).transpose(1, 2, 0)
        self.diff = inp - tiled_mb_mean
        self.mb_var = np.einsum('tbd->tb', self.diff**2)*self.normalization
        self.sqrt_var = np.sqrt(self.mb_var + self.eps)
        self.tiled_sqrt_var = np.tile(self.sqrt_var, (self.depth, 1, 1)).transpose(1, 2, 0)
        self.norm_x = self.diff/self.tiled_sqrt_var
        output = self.W*self.norm_x +self.b

        return output


    def timestep_forward(self, inp):
        self.use_timestep_mode = True
        self.timestep_backprop_init = False
        self.batch = inp.shape[0]
        self.normalization = 1./(self.depth)
        this_t_mb_mean = np.einsum('bd->b', inp)*self.normalization
        tiled_t_mb_mean = np.tile(this_t_mb_mean, (self.depth, 1)).transpose(1, 0)
        this_t_diff = inp - tiled_t_mb_mean
        this_t_mb_var = np.einsum('bd->b', this_t_diff**2)*self.normalization
        this_t_sqrt_var = np.sqrt(this_t_mb_var + self.eps)
        tiled_this_t_sqrt_var = np.tile(this_t_sqrt_var, (self.depth, 1)).transpose(1, 0)
        this_t_norm_x = this_t_diff/tiled_this_t_sqrt_var
        output = self.W*this_t_norm_x +self.b
        self.timestep_mb_mean.append(this_t_mb_mean)
        self.timestep_diff.append(this_t_diff)
        self.timestep_sqrt_var.append(this_t_sqrt_var)
        self.timestep_norm_x.append(this_t_norm_x)
        self.counter += 1

        return output

    def backprop(self, dLoss):
        if self.use_timestep_mode == True:
            self.timestep_gather()
        self.db = np.einsum('tbd->d', dLoss)
        self.dW = np.einsum('tbd->d', dLoss*self.norm_x)
        std_inv = 1./self.sqrt_var
        d_norm_x = dLoss*self.W
        d_mb_var = -0.5*np.einsum('tbd->tb', d_norm_x*self.diff)*std_inv**3
        d_mb_mean = np.einsum('tbd,tb->tb', -1*d_norm_x, std_inv) + \
            d_mb_var*np.einsum('tbd->tb', -2*self.diff)*self.normalization
        dL_prev = np.einsum('tbd,tb->tbd', d_norm_x, std_inv) +\
                  np.einsum('tb,tbd->tbd', d_mb_var, self.diff*2)*self.normalization +\
                  np.tile(d_mb_mean, (self.depth, 1, 1)).transpose(1, 2, 0)*self.normalization
        return dL_prev

    def timestep_backprop(self, dLoss):
        if not self.timestep_backprop_init:
            self.timestep_gather()
            self.timestep_backprop_init = True
            self.db = np.einsum('bd->d', dLoss)
            self.dW = np.einsum('bd->d', dLoss*self.norm_x[self.counter_backprop])
        if self.timestep_backprop_init:
            self.db += np.einsum('bd->d', dLoss)
            self.dW += np.einsum('bd->d', dLoss*self.norm_x[self.counter_backprop])
        std_inv = 1./self.sqrt_var[self.counter_backprop]
        d_norm_x = dLoss*self.W
        this_diff = self.diff[self.counter_backprop]
        d_mb_var = -0.5*np.einsum('bd->b', d_norm_x*this_diff)*std_inv**3
        d_mb_mean = np.einsum('bd,b->b', -1*d_norm_x, std_inv) + \
            d_mb_var*np.einsum('bd->b', -2*this_diff)*self.normalization
        dL_prev = np.einsum('bd,b->bd', d_norm_x, std_inv) +\
                  np.einsum('b,bd->bd', d_mb_var, this_diff*2)*self.normalization +\
                  np.tile(d_mb_mean, (self.depth, 1)).transpose(1, 0)*self.normalization
        self.counter_backprop -= 1
        return dL_prev

    def timestep_gather(self):
        self.mb_mean = np.array(self.timestep_mb_mean)
        self.diff = np.array(self.timestep_diff)
        self.sqrt_var = np.array(self.timestep_sqrt_var)
        self.norm_x = np.array(self.timestep_norm_x)
        self.timestep_mb_mean = []
        self.timestep_diff = []
        self.timestep_sqrt_var = []
        self.timestep_norm_x = []
        self.counter_backprop = self.counter - 1
        self.counter = 0

