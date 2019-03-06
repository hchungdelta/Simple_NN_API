#!/usr/bin/env python3
# -*- coding: utf_8 -*-

'''
Title : lstmhelper
Description : some function to support LSTM_layer.
'''
import numpy as np

class Sigmoid():
    def __init__(self, smooth=1):
        self.smooth = smooth
    def forward(self, inp_layer):
        inp_layer = np.where(inp_layer > 5, 5, inp_layer)
        self.sigmoid = 1/(1+np.exp(-1*inp_layer*(1./self.smooth)))
        return self.sigmoid
    def backprop(self):
        return self.sigmoid*(1-self.sigmoid)/self.smooth

class Tanh():
    def __init__(self, upperlimit=1, smooth=1):
        self.upperlimit = upperlimit
        self.smooth = smooth
    def forward(self, inp_layer):
        inp_layer = np.where(inp_layer > 5, 5, inp_layer)
        exp_term = np.exp(inp_layer*(2./self.smooth))
        self.tanh = self.upperlimit*(exp_term-1)/(exp_term+1)
        return self.tanh
    def backprop(self):
        return (1-self.tanh*self.tanh)/self.smooth
