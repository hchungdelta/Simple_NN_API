import numpy as np

class Sigmoid():
    """
    return 1/(1+e^(-x))
    """
    def __init__(self):
        self.sigmoid = None

    def forward(self, inp_layer):
        inp_layer = inp_layer

        inp_layer = np.where(inp_layer > 5, 5, inp_layer)
        self.sigmoid = 1/(1+np.exp(inp_layer*-1))
        return self.sigmoid
    def backprop(self, dL):
        return  dL*self.sigmoid*(1-self.sigmoid)

class Tanh():
    """
    return (e^(x)-e^(-x))/(e^(x)+e^(-x))
    """
    def __init__(self, upperlimit=1, smooth=1):
        self.upperlimit = upperlimit
        self.smooth = smooth
        self.tanh = None

    def forward(self, inp_layer):
        inp_layer = inp_layer
        # safe lock
        inp_layer = np.where(inp_layer > 5, 5, inp_layer)
        self.tanh = self.upperlimit*(np.exp(inp_layer*(2./self.smooth))-1)\
        /(np.exp(inp_layer*(2./self.smooth))+1)
        return self.tanh
    def backprop(self, dL):
        return dL*(1-self.tanh*self.tanh)*(1/self.smooth)



class partial_Tanh():
    """
    only tanh a certain part of input_layer
    return (e^(x)-e^(-x))/(e^(x)+e^(-x))
    """
    def __init__(self, cut_at, upperlimit=1, smooth=1):
        self.cut_at = cut_at
        self.upperlimit = upperlimit
        self.smooth = smooth
        self.tanh = None
    def forward(self, inp_layer):
        front_part = inp_layer[:, :, :self.cut_at]
        back_part = inp_layer[:, :, self.cut_at:]
        back_part = np.where(back_part > 5, 5, back_part)

        exp_term = np.exp(back_part*(2./self.smooth))
        self.tanh = self.upperlimit*(exp_term-1)/(exp_term+1)
        output = np.concatenate((front_part, self.tanh), axis=2)
        return output

    def backprop(self, dL):
        front_dL = dL[:, :, :self.cut_at]
        back_dL = dL[:, :, self.cut_at:]*(1-self.tanh*self.tanh)*(1/self.smooth)
        prev_dL = np.concatenate((front_dL, back_dL), axis=2)
        return prev_dL
