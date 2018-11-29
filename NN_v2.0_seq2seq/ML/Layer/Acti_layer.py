import numpy as np

class ReLU():
    """
    return np.max(input, 0)
    """
    def forward(self, inp_layer):
        output = np.zeros_like(inp_layer)
        output = np.maximum.reduce([inp_layer, output])
        self.output = output
        return output
    def backprop(self, dL):
        return dL*np.where(self.output != 0, 1, 0)

class Sigmoid():
    """
    return 1/(1+e^(-x))
    """
    def forward(self, inp_layer):
        self.inp_layer = inp_layer
        self.sigmoid = 1/(1+np.exp(self.inp_layer*-1))
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
    def forward(self, inp_layer):
        self.inp_layer = inp_layer
        self.tanh = self.upperlimit*(np.exp(self.inp_layer*(2./self.smooth))-1)\
        /(np.exp(self.inp_layer*(2./self.smooth))+1)
        return self.tanh
    def backprop(self, dL):
        return dL*(1-self.tanh*self.tanh)*(1/self.smooth)
