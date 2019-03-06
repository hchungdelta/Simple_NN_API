import random
import numpy as np

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
