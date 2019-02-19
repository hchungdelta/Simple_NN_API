#!/usr/bin/env python3
# -*- coding: utf_8 -*-

"""
LSTM (Long short term memory, Neural Computation 9(8) 1735-1780. (1997))
"""
import numpy as np
from ML.NN.Tools import orthogonal_initializer
from numba import jit
from ML.Layer.lstmhelper import Sigmoid, Tanh
from ML.Layer.Normalization import LSTM_LayerNorm
class LSTMcell:
    """
    Hold a group of LSTM
    Note that weights/bias must be global.
    While sigmoid/dsigmoid/dLoss ...etc must be local.
    Hence it can easily and efficiently performes forward & backward propagation.
    """
    def __init__(self, input_info, output_form="All",
                 ortho=False, LSTM_LN=False, smooth=1, dtype=np.float32):
        '''
        input_info : [length, batchsize, input depth, hidden state depth]
        output_form : All(many-to-many)/One(many-to-one)/None
        ortho: whether to switch on orthogonal_initializer
        LSTM_LN: False/1/2 - False/0: switch off inner layer normalization.
                                   1: add layer normalization. (only one)
                                   2: add layer normalization. (full mode)
        '''
        self.length = input_info[0]  # length of LSTM
        self.batch = input_info[1]
        self.input_depth = input_info[2] # length of input's depth ( or embedding length
        self.hidden_units = input_info[3] # hidden state's depth
        self.total_depth = self.hidden_units+self.input_depth  # concatenate
        self.output_form = output_form
        self.cutoff_length = self.length
        self.smooth = smooth
        # (sigmoid)forget gate/ (sigmoid)input gate / (sigmoid)output gate / (tanh)
        self.W = np.random.randn(4*self.total_depth, self.hidden_units)/np.sqrt(self.total_depth/2.)
        if ortho:
            self.W = orthogonal_initializer(self.W)
        self.b = np.random.randn(4*self.hidden_units)/np.sqrt(self.total_depth/2.)
        self.W = self.W.astype(dtype)
        self.b = self.b.astype(dtype)
        self.dtype = dtype
        self.LSTM_LN = LSTM_LN if LSTM_LN != 0 else False

        self.share_mem = [self.batch, self.input_depth, self.hidden_units,
                          self.total_depth, self.W, self.b, self.smooth, dtype, LSTM_LN]


        if self.LSTM_LN == 1:
            self.LSTM_LN_out = LSTM_LayerNorm((self.hidden_units))
            self.share_mem.extend([self.LSTM_LN_out])

        if self.LSTM_LN == 2:
            self.LSTM_LN_f_hid = LSTM_LayerNorm((self.hidden_units))
            self.LSTM_LN_f_inp = LSTM_LayerNorm((self.hidden_units))
            self.LSTM_LN_i_hid = LSTM_LayerNorm((self.hidden_units))
            self.LSTM_LN_i_inp = LSTM_LayerNorm((self.hidden_units))
            self.LSTM_LN_o_hid = LSTM_LayerNorm((self.hidden_units))
            self.LSTM_LN_o_inp = LSTM_LayerNorm((self.hidden_units))
            self.LSTM_LN_c_hid = LSTM_LayerNorm((self.hidden_units))
            self.LSTM_LN_c_inp = LSTM_LayerNorm((self.hidden_units))
            self.LSTM_LN_out = LSTM_LayerNorm((self.hidden_units))
            self.share_mem.extend([self.LSTM_LN_f_hid, self.LSTM_LN_f_inp,
                                   self.LSTM_LN_i_hid, self.LSTM_LN_i_inp,
                                   self.LSTM_LN_o_hid, self.LSTM_LN_o_inp,
                                   self.LSTM_LN_c_hid, self.LSTM_LN_c_inp,
                                   self.LSTM_LN_out])
        self.LSTMsequence = []
        for _ in range(self.length):
            self.LSTMsequence.append(LSTM(self.share_mem))
        self.LSTM_count_idx = 0 # for infer mode, record which LSTM is processing.
        self.sum_dW = np.zeros_like(self.W)
        self.sum_db = np.zeros_like(self.b)

    def rewrite_Wb(self, W, b):
        if self.LSTM_LN == False:
            self.W = W
            self.b = b

        if self.LSTM_LN == 1:
            self.W = W[0]
            self.b = b[0]
            self.LSTM_LN_out.rewrite_Wb(W[1], b[1])

        if self.LSTM_LN == 2:
            self.W = W[0]
            self.b = b[0]
            self.LSTM_LN_f_hid.rewrite_Wb(W[1], b[1])
            self.LSTM_LN_f_inp.rewrite_Wb(W[2], b[2])
            self.LSTM_LN_i_hid.rewrite_Wb(W[3], b[3])
            self.LSTM_LN_i_inp.rewrite_Wb(W[4], b[4])
            self.LSTM_LN_o_hid.rewrite_Wb(W[5], b[5])
            self.LSTM_LN_o_inp.rewrite_Wb(W[6], b[6])
            self.LSTM_LN_c_hid.rewrite_Wb(W[7], b[7])
            self.LSTM_LN_c_inp.rewrite_Wb(W[8], b[8])
            self.LSTM_LN_out.rewrite_Wb(W[9], b[9])

        for this_LSTM in self.LSTMsequence:
            this_LSTM.rewrite_Wb(self.W, self.b)

        self.sum_dW = np.zeros_like(self.W)
        self.sum_db = np.zeros_like(self.b)

    def update(self, dW, db, lr):
        if self.LSTM_LN == False:
            self.W = self.W-lr*dW
            self.b = self.b-lr*db
        if self.LSTM_LN == 1:
            self.W = self.W-lr*dW[0][0]
            self.b = self.b-lr*db[0][0]
            self.LSTM_LN_out.update(dW[0][1], db[0][1], lr)

        if self.LSTM_LN == 2:
            self.W = self.W-lr*dW[0][0]
            self.b = self.b-lr*db[0][0]
            self.LSTM_LN_f_hid.update(dW[0][1], db[0][1], lr)
            self.LSTM_LN_f_inp.update(dW[0][2], db[0][2], lr)
            self.LSTM_LN_i_hid.update(dW[0][3], db[0][3], lr)
            self.LSTM_LN_i_inp.update(dW[0][4], db[0][4], lr)
            self.LSTM_LN_o_hid.update(dW[0][5], db[0][5], lr)
            self.LSTM_LN_o_inp.update(dW[0][6], db[0][6], lr)
            self.LSTM_LN_c_hid.update(dW[0][7], db[0][7], lr)
            self.LSTM_LN_c_inp.update(dW[0][8], db[0][8], lr)
            self.LSTM_LN_out.update(dW[0][9], db[0][9], lr)
        for this_LSTM in self.LSTMsequence:
            this_LSTM.rewrite_Wb(self.W, self.b)

        self.sum_dW = np.zeros_like(self.W)
        self.sum_db = np.zeros_like(self.b)

    def forward(self, inputdata, prev_h_state, prev_c_state, cutoff_length=None):
        '''
        forward propagation:
        input:
            1. inputdata
            2. prev_h_state (if no, input None)
            3. prev_c_state (if no, input None)
            4. cutoff_length(optional): to set an upper limit length
                                        for forward propagation.
        return
            1. prev_h_list : all h state output list
            2. prev_c_list : all c state output list
            3. output_h_state : final h state
            4. output_c_state : final c state
        '''
        if cutoff_length != None:
            self.cutoff_length = cutoff_length

        if not isinstance(inputdata, np.ndarray):
            inputdata = np.zeros((self.length, self.batch, self.input_depth)).astype(self.dtype)
        prev_h_list = []
        prev_c_list = []
        for idx, this_LSTM in enumerate(self.LSTMsequence[:self.cutoff_length]):
            prev_h_state, prev_c_state = this_LSTM.forward(inputdata[idx],
                                                           prev_h_state,
                                                           prev_c_state)
            prev_h_list.append(prev_h_state[0])
            prev_c_list.append(prev_c_state[0])
        return  np.array(prev_h_list), np.array(prev_c_list), prev_h_state, prev_c_state

    def timestep_forward(self, inputdata, prev_h_state, prev_c_state, cutoff_length=None):

        if not isinstance(inputdata, np.ndarray):
            inputdata = np.zeros((self.batch, self.input_depth)).astype(self.dtype)

        this_LSTM = self.LSTMsequence[self.LSTM_count_idx]
        self.LSTM_count_idx += 1
        prev_h_state, prev_c_state = this_LSTM.forward(inputdata,
                                                       prev_h_state,
                                                       prev_c_state)

        return prev_h_state[0], prev_c_state[0], prev_h_state, prev_c_state

    def backprop(self, dLoss_list, next_dh, next_dc, cutoff_length=None):
        '''
        backward propagation:
        input:
            1. dLoss_list (if no, input None)
            2. next_dh (if no, input None)
            3. next_dc (if no, input None)
            4. cutoff_length(optional): to set an upper limit length
                                        for backward propagation.
        return
            1. local_dInput_list : all h state output list
            2. next_dh : previous dh
            3. next_dc : previous dc
        '''
        self.LSTM_count_idx = 0
        if cutoff_length != None:
            self.cutoff_length = cutoff_length

        if self.output_form == "One":
            zero_add = np.zeros_like(dLoss_list)
            dLoss_list = np.concatenate((
                np.tile(zero_add, (len(self.LSTMsequence)-1, 1, 1)),
                dLoss_list), axis=0)

        if self.output_form == "None":
            dLoss_list = [None] * len(self.LSTMsequence)

        local_dInput_list = []
        for idx, this_LSTM in enumerate(reversed(self.LSTMsequence[:self.cutoff_length])):
            local_dInput, next_dh, next_dc = this_LSTM.backprop(
                dLoss_list[len(dLoss_list)-(idx+1)],
                next_dh=next_dh,
                next_dc=next_dc)
            # get local_dW / db,  need to sum up to update the W and b.
            local_dW, local_db = this_LSTM.get_dWb()
            self.sum_dW += local_dW/(self.batch)
            self.sum_db += local_db/(self.batch)

            #self.sum_update(local_dW,local_db )
            local_dInput_list.append(local_dInput[0])
        return  np.array(local_dInput_list), next_dh, next_dc

    def get_dWb(self):
        if self.LSTM_LN == 1:
            dout_dW, dout_db = self.LSTM_LN_out.get_dWb()
            dWs = [self.sum_dW, dout_dW]
            dbs = [self.sum_db, dout_db]

        if self.LSTM_LN == 2:
            f_hid_dW, f_hid_db = self.LSTM_LN_f_hid.get_dWb()
            f_inp_dW, f_inp_db = self.LSTM_LN_f_inp.get_dWb()
            i_hid_dW, i_hid_db = self.LSTM_LN_i_hid.get_dWb()
            i_inp_dW, i_inp_db = self.LSTM_LN_i_inp.get_dWb()
            o_hid_dW, o_hid_db = self.LSTM_LN_o_hid.get_dWb()
            o_inp_dW, o_inp_db = self.LSTM_LN_o_inp.get_dWb()
            c_hid_dW, c_hid_db = self.LSTM_LN_c_hid.get_dWb()
            c_inp_dW, c_inp_db = self.LSTM_LN_c_inp.get_dWb()

            dout_dW, dout_db = self.LSTM_LN_out.get_dWb()
            dWs = [self.sum_dW, f_hid_dW, f_inp_dW, i_hid_dW, i_inp_dW, o_hid_dW, o_inp_dW, c_hid_dW, c_inp_dW, dout_dW]
            dbs = [self.sum_db, f_hid_db, f_inp_db, i_hid_db, i_inp_db, o_hid_db, o_inp_db, c_hid_db, c_inp_db, dout_db]
        return (self.sum_dW, self.sum_db) if self.LSTM_LN == False else (dWs, dbs)

    def get_Wb(self):
        if self.LSTM_LN == 1:
            out_W, out_b = self.LSTM_LN_out.get_Wb()
            Ws = [self.W, out_W]
            bs = [self.b, out_b]

        if self.LSTM_LN == 2:
            f_hid_W, f_hid_b = self.LSTM_LN_f_hid.get_Wb()
            f_inp_W, f_inp_b = self.LSTM_LN_f_inp.get_Wb()
            i_hid_W, i_hid_b = self.LSTM_LN_i_hid.get_Wb()
            i_inp_W, i_inp_b = self.LSTM_LN_i_inp.get_Wb()
            o_hid_W, o_hid_b = self.LSTM_LN_o_hid.get_Wb()
            o_inp_W, o_inp_b = self.LSTM_LN_o_inp.get_Wb()
            c_hid_W, c_hid_b = self.LSTM_LN_c_hid.get_Wb()
            c_inp_W, c_inp_b = self.LSTM_LN_c_inp.get_Wb()

            out_W, out_b = self.LSTM_LN_out.get_Wb()
            Ws = [self.W, f_hid_W, f_inp_W, i_hid_W, i_inp_W, o_hid_W, o_inp_W, c_hid_W, c_inp_W, out_W]
            bs = [self.b, f_hid_b, f_inp_b, i_hid_b, i_inp_b, o_hid_b, o_inp_b, c_hid_b, c_inp_b, out_b]
        return (self.W, self.b) if self.LSTM_LN == False else (Ws, bs)

    def timestep_gather(self):
        self.LSTM_count_idx = 0


class LSTM():
    def __init__(self, share_mem):
        # global
        self.batch = share_mem[0]
        self.input_depth = share_mem[1]
        self.hidden_units = share_mem[2]
        self.total_depth = share_mem[3]
        self.W = share_mem[4]
        self.b = share_mem[5]
        self.smooth = share_mem[6]
        self.dtype = share_mem[7]
        self.LSTM_LN_on = share_mem[8]
        if  self.LSTM_LN_on == 1:
            self.LSTM_LN_out = share_mem[9]

        if  self.LSTM_LN_on == 2:
            self.LSTM_LN_f_hid = share_mem[9]
            self.LSTM_LN_f_inp = share_mem[10]
            self.LSTM_LN_i_hid = share_mem[11]
            self.LSTM_LN_i_inp = share_mem[12]
            self.LSTM_LN_o_hid = share_mem[13]
            self.LSTM_LN_o_inp = share_mem[14]
            self.LSTM_LN_c_hid = share_mem[15]
            self.LSTM_LN_c_inp = share_mem[16]
            self.LSTM_LN_out = share_mem[17]

            self.LSTM_LNs = [(self.LSTM_LN_f_hid, self.LSTM_LN_f_inp), 
                             (self.LSTM_LN_i_hid, self.LSTM_LN_i_inp),
                             (self.LSTM_LN_o_hid, self.LSTM_LN_o_inp),
                             (self.LSTM_LN_c_hid, self.LSTM_LN_c_inp)]
        # local
        self.Sigmoid = Sigmoid(smooth=self.smooth)
        self.Tanhc = Tanh(smooth=self.smooth)
        self.Tanho = Tanh(smooth=self.smooth)
        self.local_dW = np.zeros_like(self.W)
        self.local_db = np.zeros_like(self.b)
        self.empty_state = np.zeros((self.batch, self.hidden_units)).astype(self.dtype)
    @jit
    def forward(self, inp_layer, prev_state_h, prev_state_c):
        if not isinstance(inp_layer, np.ndarray):
            inp_layer = np.zeros((self.batch, self.input_depth)).astype(self.dtype)
        self.inp_layer = inp_layer
        if not isinstance(prev_state_h, np.ndarray) and not isinstance(prev_state_c, np.ndarray):
            self.prev_c = self.empty_state
            self.prev_h = self.empty_state
        else:
            self.prev_c = prev_state_c[0]
            self.prev_h = prev_state_h[0]
        self.state = np.concatenate([self.prev_h, self.inp_layer], axis=1)
        xW = np.zeros((self.batch, 4*self.hidden_units)).astype(self.dtype)


        for inner_idx in range(4):
             intv_start = inner_idx*self.total_depth
             intv_mid = intv_start + self.hidden_units
             intv_end = intv_start + self.total_depth
             xW_start = inner_idx*self.hidden_units
             xW_end = (inner_idx+1)*self.hidden_units
             hidden_term = np.matmul(self.state[:, :self.hidden_units], self.W[intv_start:intv_mid])
             input_term = np.matmul(self.state[:, self.hidden_units:], self.W[intv_mid:intv_end])
             
             if self.LSTM_LN_on == 2:
                 hidden_term = self.LSTM_LNs[inner_idx][0].timestep_forward(hidden_term)
                 input_term = self.LSTM_LNs[inner_idx][1].timestep_forward(input_term)   
             xW[:, xW_start: xW_end] += hidden_term 
             xW[:, xW_start: xW_end] += input_term
             #xW[:, inner_idx*self.hidden_units:(inner_idx+1)*self.hidden_units] = np.matmul(
             #    self.state, self.W[inner_idx*self.total_depth:(inner_idx+1)*self.total_depth]) 

        self.xW_b = xW + self.b
        sig_xW_b = self.Sigmoid.forward(self.xW_b[:, :3*self.hidden_units])

        self.Hf = sig_xW_b[:, 0*self.hidden_units :1*self.hidden_units]  # forget gate
        self.Hi = sig_xW_b[:, 1*self.hidden_units :2*self.hidden_units]  # input gate
        self.Ho = sig_xW_b[:, 2*self.hidden_units :3*self.hidden_units]  # output gate
        self.Hc = self.Tanhc.forward(self.xW_b[:, 3*self.hidden_units:]) # candidate state

        self.c = self.Hf*self.prev_c+self.Hi*self.Hc   # renew state
        _c =  self.LSTM_LN_out.timestep_forward(self.c) if self.LSTM_LN_on != False else self.c
        self.Tanh_o = self.Tanho.forward(_c)       # candidate output
        self.h = self.Ho*self.Tanh_o                  # output  (candidate output * sigmoid)

        # return output / next state
        return np.array([self.h]), np.array([self.c])
    @jit
    def backprop(self, dLoss, next_dh, next_dc):
        # derivative of tanh and sigmoid
        dTanh_c = self.Tanhc.backprop()
        dTanh_o = self.Tanho.backprop()
        dSigmoid = self.Sigmoid.backprop()  # f/i/o   //c
        # section of Hf/Hi/Ho
        dSigmoid_Hf = dSigmoid[:, 0*self.hidden_units :1*self.hidden_units]
        dSigmoid_Hi = dSigmoid[:, 1*self.hidden_units :2*self.hidden_units]
        dSigmoid_Ho = dSigmoid[:, 2*self.hidden_units :3*self.hidden_units]


        if not isinstance(next_dh, np.ndarray) and not isinstance(next_dc, np.ndarray):
            next_dc = self.empty_state
            next_dh = self.empty_state
        else:
            next_dc = next_dc[0]
            next_dh = next_dh[0]
        if isinstance(dLoss, np.ndarray):
            dL = dLoss+next_dh
            _dh_part = dL*self.Ho*dTanh_o
            dh_part = self.LSTM_LN_out.timestep_backprop(_dh_part) if self.LSTM_LN_on != False else _dh_part
            dc = dh_part + next_dc
            dbo = dL*dSigmoid_Ho*self.Tanh_o     #output gate
        else:
            dL = next_dh
            _dh_part = dL*self.Ho*dTanh_o
            dh_part = self.LSTM_LN_out.timestep_backprop(_dh_part) if self.LSTM_LN_on != False else _dh_part
            dc = dh_part + next_dc
            dbo = dL*dSigmoid_Ho*self.Tanh_o    #output gate
        dbf = dc*dSigmoid_Hf*self.prev_c       # forget gate
        dbi = dc*dSigmoid_Hi*self.Hc           # input gate
        dbc = dc*dTanh_c*self.Hi           # candidate state

        db = np.concatenate((dbf, dbi, dbo, dbc), axis=1)
        self.local_db = np.sum(db, axis=0)
        if  self.LSTM_LN_on == 2:
            dbf_hid = self.LSTM_LN_f_hid.timestep_backprop(dbf)
            dbf_inp = self.LSTM_LN_f_inp.timestep_backprop(dbf)
            dbi_hid = self.LSTM_LN_i_hid.timestep_backprop(dbi)
            dbi_inp = self.LSTM_LN_i_inp.timestep_backprop(dbi)
            dbo_hid = self.LSTM_LN_o_hid.timestep_backprop(dbo)
            dbo_inp = self.LSTM_LN_o_inp.timestep_backprop(dbo)
            dbc_hid = self.LSTM_LN_c_hid.timestep_backprop(dbc)
            dbc_inp = self.LSTM_LN_c_inp.timestep_backprop(dbc)
            
            state_hid_part = self.state[:, :self.hidden_units].T
            state_inp_part = self.state[:,self.hidden_units:].T
            self.local_dW[0*self.total_depth:self.hidden_units] = np.dot(state_hid_part, dbf_hid)
            self.local_dW[self.hidden_units:1*self.total_depth] = np.dot(state_inp_part, dbf_inp)
            self.local_dW[1*self.total_depth:1*self.total_depth+self.hidden_units] = np.dot(state_hid_part, dbi_hid)
            self.local_dW[1*self.total_depth+self.hidden_units:2*self.total_depth] = np.dot(state_inp_part, dbi_inp)
            self.local_dW[2*self.total_depth:2*self.total_depth+self.hidden_units] = np.dot(state_hid_part, dbo_hid)
            self.local_dW[2*self.total_depth+self.hidden_units:3*self.total_depth] = np.dot(state_inp_part, dbo_inp)
            self.local_dW[3*self.total_depth:3*self.total_depth+self.hidden_units] = np.dot(state_hid_part, dbc_hid)
            self.local_dW[3*self.total_depth+self.hidden_units:4*self.total_depth] = np.dot(state_inp_part, dbc_inp)
            self.local_dL = np.zeros_like(self.state)
            self.local_dL[:, :self.hidden_units] += np.matmul(dbf_hid, self.W[0*self.total_depth:self.hidden_units].T)
            self.local_dL[:, self.hidden_units:] += np.matmul(dbf_inp, self.W[self.hidden_units:1*self.total_depth].T)
            self.local_dL[:, :self.hidden_units] += np.matmul(dbi_hid, self.W[1*self.total_depth:1*self.total_depth+self.hidden_units].T)
            self.local_dL[:, self.hidden_units:] += np.matmul(dbi_inp, self.W[1*self.total_depth+self.hidden_units:2*self.total_depth].T)
            self.local_dL[:, :self.hidden_units] += np.matmul(dbo_hid, self.W[2*self.total_depth:2*self.total_depth+self.hidden_units].T)
            self.local_dL[:, self.hidden_units:] += np.matmul(dbo_inp, self.W[2*self.total_depth+self.hidden_units:3*self.total_depth].T)
            self.local_dL[:, :self.hidden_units] += np.matmul(dbc_hid, self.W[3*self.total_depth:3*self.total_depth+self.hidden_units].T)
            self.local_dL[:, self.hidden_units:] += np.matmul(dbc_inp, self.W[3*self.total_depth+self.hidden_units:4*self.total_depth].T)
        else:
            self.local_dW[0*self.total_depth:1*self.total_depth] = np.dot(self.state.T, dbf)
            self.local_dW[1*self.total_depth:2*self.total_depth] = np.dot(self.state.T, dbi)
            self.local_dW[2*self.total_depth:3*self.total_depth] = np.dot(self.state.T, dbo)
            self.local_dW[3*self.total_depth:4*self.total_depth] = np.dot(self.state.T, dbc)

            self.local_dL = np.matmul(dbf, self.W[0*self.total_depth:1*self.total_depth].T)
            self.local_dL += np.matmul(dbi, self.W[1*self.total_depth:2*self.total_depth].T)
            self.local_dL += np.matmul(dbo, self.W[2*self.total_depth:3*self.total_depth].T)
            self.local_dL += np.matmul(dbc, self.W[3*self.total_depth:4*self.total_depth].T)


        #for single_data in  range(self.batch):
        next_dh = np.array([self.local_dL[:, :self.hidden_units]])     # d input
        next_dInput = np.array([self.local_dL[:, self.hidden_units:]]) # d input
        next_dc = np.array([self.Hf*dc])
        return  next_dInput, next_dh, next_dc


    def rewrite_Wb(self, W, b):
        """
        rewrite W and b. (for rank > 0 or restore)
        """
        self.W = W
        self.b = b
        self.local_dW = np.zeros_like(self.W)
        self.local_db = np.zeros_like(self.b)

    def get_Wb(self):
        """
        get W and b.
        """
        return self.W, self.b
    def get_dWb(self):
        """
        get dW and db.
        """
        return self.local_dW, self.local_db
class BiLSTM:
    '''
    concatenate forward/backward cell.
    '''
    def __init__(self, fw_Encoder, bw_Encoder, LSTM_LN=False):
        """
        1. fw_Encoder : forward encoder (as normal LSTMcell)
        2. bw_Encoder : backward encoder (as normal LSTMcell)
        """
        self.fw_Encoder = fw_Encoder
        self.bw_Encoder = bw_Encoder
        self.fw_hidden_units = fw_Encoder.hidden_units
        self.fw_total_depth = fw_Encoder.total_depth
        self.LSTM_LN = False if LSTM_LN ==0 else LSTM_LN
    def forward(self, inputdata, prev_h_state, prev_c_state, cutoff_length=None):
        """
        forward propagation:
        input:
            1. inputdata
            2. prev_h_state (is no, input None)
            3. prev_c_state (is no, input None)
        return
            1. prev_h_list : all h state output list
            2. prev_c_list : all c state output list
            3. output_h_state : final h state
            4. output_c_state : final c state
        *All in concat form (fw,bw)
        """
        if cutoff_length != None:
            reversed_inputdata = inputdata[::-1][-1*cutoff_length:]
        else:
            reversed_inputdata = inputdata[::-1]
        prev_fh_list, prev_fc_list, output_fh_state, output_fc_state = self.fw_Encoder.forward(
            inputdata, prev_h_state, prev_c_state, cutoff_length)
        prev_bh_list, prev_bc_list, output_bh_state, output_bc_state = self.bw_Encoder.forward(
            reversed_inputdata, prev_h_state, prev_c_state, cutoff_length)

        # reverse
        prev_h_list = np.concatenate((prev_fh_list, prev_bh_list[::-1]), axis=2)
        prev_c_list = np.concatenate((prev_fc_list, prev_bc_list[::-1]), axis=2)
        output_h_state = np.concatenate((output_fh_state, output_bh_state), axis=2)
        output_c_state = np.concatenate((output_fc_state, output_bc_state), axis=2)
        return prev_h_list, prev_c_list, output_h_state, output_c_state
    def backprop(self, dLoss, next_dh, next_dc, cutoff_length=None):
        """
        back propagation:
        input :
            1. dLoss   (is no, input None)
            2. next_dh (is no, input None)
            3. next_dc (is no, input None)
        return dLoss, prev_dh, prev_dc
        """
        if isinstance(dLoss, np.ndarray):
            dLoss_fw = dLoss[:, :, :self.fw_hidden_units]
            dLoss_bw = dLoss[:, :, self.fw_hidden_units:][::-1]
        else:
            dLoss_fw = dLoss  # None
            dLoss_bw = dLoss  # None
        next_fdh = next_dh[:, :, :self.fw_hidden_units]
        next_bdh = next_dh[:, :, self.fw_hidden_units:]
        next_fdc = next_dc[:, :, :self.fw_hidden_units]
        next_bdc = next_dc[:, :, self.fw_hidden_units:]

        dLoss_fw, init_fdh, init_fdc = self.fw_Encoder.backprop(
            dLoss_fw, next_fdh, next_fdc, cutoff_length)
        dLoss_bw, init_bdh, init_bdc = self.bw_Encoder.backprop(
            dLoss_bw, next_bdh, next_bdc, cutoff_length)
        init_dh = np.concatenate((init_fdh, init_bdh), axis=1)
        init_dc = np.concatenate((init_fdc, init_bdc), axis=1)

        dLoss = []
        for a in range(len(dLoss_fw)):
            dLoss.append((dLoss_fw[a] + dLoss_bw[len(dLoss_fw)-(1+a)]))
        return np.array(dLoss), init_dh, init_dc
    def update(self, dW, db, lr):
        """
        to update W and b (for rank=0)
        """
        if self.LSTM_LN == False:
            dW_fw = dW[:4*self.fw_total_depth]
            dW_bw = dW[4*self.fw_total_depth:]
            db_fw = db[:4*self.fw_hidden_units]
            db_bw = db[4*self.fw_hidden_units:]
            self.fw_Encoder.update(dW_fw, db_fw, lr)
            self.bw_Encoder.update(dW_bw, db_bw, lr)

        if self.LSTM_LN == 1 or  self.LSTM_LN == 2:
            dW = dW[0]
            db = db[0]
            dW_fws = [dW[0][:4*self.fw_total_depth]]
            dW_bws = [dW[0][4*self.fw_total_depth:]]
            db_fws = [db[0][:4*self.fw_hidden_units]]
            db_bws = [db[0][4*self.fw_hidden_units:]]
            for W_idx in range(1, len(dW)):
                dW_fws.append(dW[W_idx][:self.fw_hidden_units])
                dW_bws.append(dW[W_idx][self.fw_hidden_units:])
                db_fws.append(db[W_idx][:self.fw_hidden_units])
                db_bws.append(db[W_idx][self.fw_hidden_units:])
            self.fw_Encoder.update([dW_fws],[db_fws], lr)
            self.bw_Encoder.update([dW_bws],[db_bws], lr)

    def get_dWb(self):
        """
        get dW and db from both forward and backward encoder.
        """
        if self.LSTM_LN == False:
            dW_fw, db_fw = self.fw_Encoder.get_dWb()
            dW_bw, db_bw = self.bw_Encoder.get_dWb()
            dW = np.concatenate((dW_fw, dW_bw), axis=0)
            db = np.concatenate((db_fw, db_bw), axis=0)
        if self.LSTM_LN == 1 or self.LSTM_LN == 2:
            dW_fws, db_fws = self.fw_Encoder.get_dWb()
            dW_bws, db_bws = self.bw_Encoder.get_dWb()
            dWs = []
            dbs = []
            for W_idx in range(len(dW_fws)):
                this_dW = np.concatenate((dW_fws[W_idx], dW_bws[W_idx]), axis=0)
                this_db = np.concatenate((db_fws[W_idx], db_bws[W_idx]), axis=0)
                dWs.append(this_dW)
                dbs.append(this_db)
        return (dW, db) if self.LSTM_LN == False else (dWs, dbs)

    def get_Wb(self):
        """
        get W and b from both forward and backward encoder.
        """
        if self.LSTM_LN == False:
            W_fw, b_fw = self.fw_Encoder.get_Wb()
            W_bw, b_bw = self.bw_Encoder.get_Wb()
            W = np.concatenate((W_fw, W_bw), axis=0)
            b = np.concatenate((b_fw, b_bw), axis=0)
        if self.LSTM_LN == 1 or self.LSTM_LN == 2:
            W_fws, b_fws = self.fw_Encoder.get_Wb()
            W_bws, b_bws = self.bw_Encoder.get_Wb()
            Ws = []
            bs = []
            for W_idx in range(len(W_fws)):
                this_W = np.concatenate((W_fws[W_idx], W_bws[W_idx]), axis=0)
                this_b = np.concatenate((b_fws[W_idx], b_bws[W_idx]), axis=0)
                Ws.append(this_W)
                bs.append(this_b)
        return (W, b) if self.LSTM_LN  == False else (Ws, bs) 


    def rewrite_Wb(self, W, b):
        """
        rewrite W and b in both forward and backward encoder.
        split W into W_fw(forward) and W_bw(backward)
              b into b_fw(forward) and b_bw(backward)
        and deliver these tensor to fw_encoder and bw_encoder
        """
        if self.LSTM_LN == False:
            W_fw = W[:4*self.fw_total_depth]
            W_bw = W[4*self.fw_total_depth:]
            b_fw = b[:4*self.fw_hidden_units]
            b_bw = b[4*self.fw_hidden_units:]

            self.fw_Encoder.rewrite_Wb(W_fw, b_fw)
            self.bw_Encoder.rewrite_Wb(W_bw, b_bw)

        if self.LSTM_LN == 1 or self.LSTM_LN == 2:
            W_fw = W[0][:4*self.fw_total_depth]
            W_bw = W[0][4*self.fw_total_depth:]
            b_fw = b[0][:4*self.fw_hidden_units]
            b_bw = b[0][4*self.fw_hidden_units:]

            W_fws = [W_fw]
            W_bws = [W_bw]
            b_fws = [b_fw]
            b_bws = [b_bw]
            for W_idx in range(1, len(W)):
                LN_W_fw = W[W_idx][:self.fw_hidden_units]
                LN_W_bw = W[W_idx][self.fw_hidden_units:]
                LN_b_fw = b[W_idx][:self.fw_hidden_units]
                LN_b_bw = b[W_idx][self.fw_hidden_units:]
                W_fws.append(LN_W_fw)
                W_bws.append(LN_W_bw)
                b_fws.append(LN_b_fw)
                b_bws.append(LN_b_bw)
            self.fw_Encoder.rewrite_Wb(W_fws, b_fws)
            self.bw_Encoder.rewrite_Wb(W_bws, b_bws)





