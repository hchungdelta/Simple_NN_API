import numpy as np
from numba import njit, jit
from ML.Layer.Acti_layer import Sigmoid, Tanh

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


class BatchNorm_4d(trainable_layer):
    """
    input in shape (i, j, k, z)
    the mean/var is calculated by averging i*j*k
    """
    def __init__(self, hidden_units, mode):
        """
        create 2 vectors, weight and bias in shape hidden_units
        mode = train or infer
        """
        self.W = 1-np.random.random((hidden_units))*0.2  # equals to gamma
        self.b = np.random.random((hidden_units))*0.2  # equals to beta
        self.mode = mode
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
            self.sqrt_var = np.sqrt(self.mb_var + 1e-8)
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
        d_mb_var = -0.5*np.einsum('ftbd->d', d_norm_x*self.diff)*std_inv**3
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

class BatchNorm_heavy(trainable_layer):
    """
    still developing
    input in shape (i, j, k, z)
    the mean/var is calculated by averging k
    """
    def __init__(self, hidden_units, mode):
        """
        create 2 vectors, weight and bias in shape hidden_units
        mode = train or infer
        """
        self.W = 1-np.random.random(hidden_units)*0.2  # equals to gamma
        self.b = np.random.random(hidden_units)*0.2  # equals to beta
        self.mode = mode
        self.recent_var = []
        self.recent_mean = []
        self.record_amount = 10

    def forward(self, inp):
        self.inp_filters = inp.shape[0]
        self.timestep = inp.shape[1]
        self.batch = inp.shape[2]
        self.normalization = 1./(self.batch*self.timestep*self.inp_filters)

        if self.mode == 'train':
            # mini-batch mean
            self.mb_mean = np.mean(inp, axis=2)
            self.mb_mean_tile = np.tile(self.mb_mean, (self.batch, 1, 1, 1)).transpose(1, 2, 0, 3)
            self.diff = inp - self.mb_mean_tile
            self.mb_var = np.mean(self.diff**2, axis=2)
            self.sqrt_var = np.sqrt(self.mb_var + 1e-8)
            self.sqrt_var_tile = np.tile(self.sqrt_var, (self.batch, 1, 1, 1)).transpose(1, 2, 0, 3)
            self.norm_x = self.diff/self.sqrt_var_tile
            self.b_tile = np.tile(self.b, (self.batch, 1, 1, 1)).transpose(1, 2, 0, 3)
            output = np.einsum('ftd,ftbd->ftbd', self.W, self.norm_x) + self.b_tile
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
        self.db = np.einsum('ftbd->ftd', dLoss)
        self.dW = np.einsum('ftbd->ftd', dLoss*self.norm_x)
        std_inv = 1./self.sqrt_var_tile
        d_norm_x = np.einsum('ftd,ftbd->ftbd', self.W, dLoss)
        d_mb_var = -0.5*np.tile(np.einsum('ftbd->ftd', d_norm_x*self.diff),
                                (self.batch, 1, 1, 1)).transpose(1, 2, 0, 3)
        d_mb_var = d_mb_var*std_inv**3
        d_mb_mean = -1*np.tile(np.einsum('ftbd->ftd', d_norm_x*std_inv),
                               (self.batch, 1, 1, 1)).transpose(1, 2, 0, 3)+\
                    -2*d_mb_var*np.tile(np.mean(self.diff, axis=2),
                                        (self.batch, 1, 1, 1)).transpose(1, 2, 0, 3)
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



class BatchNorm_2d(trainable_layer):
    """
    input in shape (i, j)
    the mean/var is calculated by averging i
    """
    def __init__(self, hidden_units, mode):
        """
        create 2 vectors, weight and bias in shape hidden_units
        mode = train or infer
        """
        self.W = 1-np.random.random((hidden_units))*0.2  # equals to gamma
        self.b = np.random.random((hidden_units))*0.2  # equals to beta
        self.mode = mode
        self.recent_var = []
        self.recent_mean = []
        self.record_amount = 10
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
            self.sqrt_var = np.sqrt(self.mb_var + 1e-8)
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

class BatchNorm_scalar(trainable_layer):
    """
    W and b are only scalars.
    """
    def __init__(self, mode='train'):
        """
        create two scalars, weight and bias.
        mode = train or infer
        """
        self.W = 1.0-np.random.random(1)*0.2  # rquals to gamma
        self.b = np.random.random(1)*0.2  # equals to beta
        self.mode = mode
        self.recent_var = []
        self.recent_mean = []
        self.record_amount = 10

    def forward(self, inp):
        self.inp_filters = inp.shape[0]
        self.timestep = inp.shape[1]
        self.batch = inp.shape[2]
        self.normalization = 1./(self.batch*self.timestep*self.inp_filters)

        if self.mode == 'train':
            # mini-batch mean
            self.mb_mean = np.einsum('ftbd->d', inp)*self.normalization
            self.diff = inp - self.mb_mean
            self.mb_var = np.einsum('ftbd->d', self.diff**2)*self.normalization
            self.eps = np.ones_like(self.mb_var)*1e-8
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
        self.db = np.sum(dLoss)*self.normalization
        self.dW = np.sum(dLoss*self.norm_x)*self.normalization
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
        infer_mean = np.mean(np.array(self.recent_mean), axis=0)
        infer_var = np.mean(np.array(self.recent_var), axis=0)
        parameter = {"infer_mean":infer_mean, "infer_var":infer_var}
        return parameter

