#!/usr/bin/env python3
# -*- coding: utf_8 -*-

"""
To hold a group of LSTM
Note that weights/bias must be global.
While sigmoid/dsigmoid/dLoss ...etc must be local.
Hence it can easily and efficiently performes forward & backward propagation.
for alpha_t in range(len(alpha)):
"""
import numpy as np
from numba import jit
from ML.Layer.FCL_layer import *
from ML.Layer.lstmhelper import Sigmoid, Tanh, ThreeD_onehot
class LSTMcell:
    """
    Hold a group of LSTM
    Note that weights/bias must be global.
    While sigmoid/dsigmoid/dLoss ...etc must be local.
    Hence it can easily and efficiently performes forward & backward propagation.
    """
    def __init__(self, input_info, output_form="All", smooth=1):
        '''
        input_info : [length, batchsize, input depth, hidden state depth]
        output_form : All(many-to-many)/One(many-to-one)/None
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
        self.b = np.random.randn(4*self.hidden_units)/np.sqrt(self.total_depth/2.)


        self.share_mem = [self.batch, self.input_depth, self.hidden_units,
                          self.total_depth, self.W, self.b, self.smooth]
        self.LSTMsequence = []
        self.build_sequence()

        self.sum_dW = np.zeros_like(self.W)
        self.sum_db = np.zeros_like(self.b)

    def rewrite_Wb(self, W, b):
        self.W = W
        self.b = b
        for this_LSTM in self.LSTMsequence:
            this_LSTM.rewrite_Wb(self.W, self.b)
        self.sum_dW = np.zeros_like(self.W)
        self.sum_db = np.zeros_like(self.b)

    def update(self, dW, db, lr):
        self.W = self.W-lr*dW
        self.b = self.b-lr*db
        for this_LSTM in self.LSTMsequence:
            this_LSTM.rewrite_Wb(self.W, self.b)
        self.sum_dW = np.zeros_like(self.W)
        self.sum_db = np.zeros_like(self.b)

    def sum_update(self, sum_dW, sum_db):
        self.sum_dW += sum_dW
        self.sum_db += sum_db

    def build_sequence(self):
        """
        build a sequence of LSTMs.
        """
        for _ in range(self.length):
            self.LSTMsequence.append(LSTM(self.share_mem))
    def forward(self, inputdata, prev_h_state, prev_c_state, cutoff_length=None):
        '''
        forward propagation:
        input:
            1. inputdata
            2. prev_h_state (is no, input None)
            3. prev_c_state (is no, input None)
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
            inputdata = np.zeros((self.length, self.batch, self.input_depth))
        prev_h_list = []
        prev_c_list = []
        for idx, this_LSTM in enumerate(self.LSTMsequence[:self.cutoff_length]):
            prev_h_state, prev_c_state = this_LSTM.forward(inputdata[idx],
                                                           prev_h_state,
                                                           prev_c_state)
            prev_h_list.append(prev_h_state[0])
            prev_c_list.append(prev_c_state[0])
        return  np.array(prev_h_list), np.array(prev_c_list), prev_h_state, prev_c_state

    def backprop(self, dLoss_list, next_dh, next_dc, cutoff_length=None):
        '''
        backward propagation:
        input:
            1. dLoss_list (is no, input None)
            2. next_dh (is no, input None)
            3. next_dc (is no, input None)
            4. cutoff_length(optional): to set an upper limit length
                                        for backward propagation.
        return
            1. local_dInput_list : all h state output list
            2. next_dh : previous dh
            3. next_dc : previous dc
        '''
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
            self.sum_dW += local_dW/(self.length*self.batch)
            self.sum_db += local_db/(self.length*self.batch)
            #self.sum_update(local_dW,local_db )
            local_dInput_list.append(local_dInput[0])

        return  local_dInput_list, next_dh, next_dc
    def get_dWb(self):
        return self.sum_dW, self.sum_db
    def get_Wb(self):
        return self.W, self.b

class InferLSTMcell():
    """
    In order to perform 'infer' mode, which cannot be done efficiently in layer-wise approach
    already incorporate the softmax function and the FCL layer before
    """
    def __init__(self, LSTMcell_list, loss_function="softmax"):
        self.loss_function = loss_function
        self.LSTMcell_list = LSTMcell_list
        self.amount_of_layer = len(LSTMcell_list)
    def forward(self, encoder_states, Outputlayer, W2Vlayer,
                target, cutoff_length, attention_mode=False, attention_wrapper=None):
        """
        encoder_states : in shape [[h1,h2],[c1,c2]]
        Outputlayer : For output, decoder fully connected layer
        W2Vlayer : word2vec (embeddings) layer, for next decoder input.
        """
        self.cutoff_length = cutoff_length

        prev_h_state = encoder_states[0]
        prev_c_state = encoder_states[1]
        input_state = None
        # softmax : to gather all information throughout the whole timesteps
        pred = []
        L = 0.
        All_dLoss = []
        for timestep in range(self.cutoff_length):
            for idx, LSTMcell in enumerate(self.LSTMcell_list):
                prev_h_state[idx], prev_c_state[idx] = LSTMcell.LSTMsequence[timestep].forward(
                    input_state, prev_h_state[idx], prev_c_state[idx])
                if idx < self.amount_of_layer-1:
                    input_state = prev_h_state[idx][0]
                else:
                    if attention_mode:
                        attention_mechansim_function = attention_wrapper[0]
                        en_h_list = attention_wrapper[1]
                        attentioned_prev_h_state = attention_mechansim_function.forward(
                            en_h_list, prev_h_state[idx])
                        timepiece_output = Outputlayer.timepiece_forward(attentioned_prev_h_state)

                    if not attention_mode:
                        timepiece_output = Outputlayer.timepiece_forward(prev_h_state[idx])

                    if self.loss_function == "softmax":
                        this_pred, this_L, this_dLoss = timestep_softmax_cross_entropy(
                            timepiece_output, target[timestep])
                        pred.append(this_pred)
                        next_input_state_before_vectorized = ThreeD_onehot(
                            np.argmax(this_pred, axis=2), this_pred.shape[2])
                        input_state = W2Vlayer.just_forward(next_input_state_before_vectorized)[0]

                    if self.loss_function == "square":
                        this_L, this_dLoss = timestep_square_loss(timepiece_output,
                                                                  target[timestep])
                        pred.append(timepiece_output)
                        input_state = timepiece_output[0]

                    L += this_L
                    All_dLoss.append(this_dLoss)

        # gather all timestep_x, in order to backprop together.
        Outputlayer.timepiece_gather()
        return np.squeeze(np.array(pred), axis=1), L, np.squeeze(np.array(All_dLoss), axis=1)
    def just_forward(self, encoder_states, Outputlayer, W2Vlayer,
                     cutoff_length, attention_mode=False, attention_wrapper=None):
        # encoder_states =  [[h1,h2],[c1,c2]]
        self.cutoff_length = cutoff_length
        prev_h_state = encoder_states[0]
        prev_c_state = encoder_states[1]
        input_state = None
        # softmax : to gather all information throughout the whole timesteps
        pred = []
        for timestep in range(self.cutoff_length):
            for  idx, LSTMcell in enumerate(self.LSTMcell_list):
                prev_h_state[idx], prev_c_state[idx] = LSTMcell.LSTMsequence[timestep].forward(
                    input_state, prev_h_state[idx], prev_c_state[idx])
                if idx < self.amount_of_layer-1:
                    input_state = prev_h_state[idx][0]
                else:
                    if attention_mode:
                        attention_mechansim_function = attention_wrapper[0]
                        en_h_list = attention_wrapper[1]
                        attentioned_prev_h_state = attention_mechansim_function.forward(
                            en_h_list, prev_h_state[idx])
                        timepiece_output = Outputlayer.timepiece_forward(attentioned_prev_h_state)
                    if not attention_mode:
                        timepiece_output = Outputlayer.timepiece_forward(prev_h_state[idx])
                    if self.loss_function == "softmax":
                        this_pred = timestep_softmax(timepiece_output)
                        pred.append(this_pred)
                        next_input_state_before_vectorized = ThreeD_onehot(
                            np.argmax(this_pred, axis=2), this_pred.shape[2])
                        input_state = W2Vlayer.just_forward(next_input_state_before_vectorized)[0]

        Outputlayer.timepiece_gather()
        return np.squeeze(np.array(pred), axis=1)

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
        # local
        self.Sigmoid = Sigmoid(smooth=self.smooth)
        self.Tanhc = Tanh(smooth=self.smooth)
        self.Tanho = Tanh(smooth=self.smooth)
        self.local_dW = np.zeros_like(self.W)
        self.local_db = np.zeros_like(self.b)
        self.empty_state = np.zeros((self.batch, self.hidden_units))
    # LSTM #
    @jit
    def forward(self, inp_layer, prev_state_h, prev_state_c):
        if not isinstance(inp_layer, np.ndarray):
            inp_layer = np.zeros((self.batch, self.input_depth))
        self.inp_layer = inp_layer
        if not isinstance(prev_state_h, np.ndarray) and not isinstance(prev_state_c, np.ndarray):
            self.prev_c = self.empty_state
            self.prev_h = self.empty_state
        else:
            self.prev_c = prev_state_c[0]
            self.prev_h = prev_state_h[0]
        # concatenate current input & previous state
        self.state = np.concatenate([self.prev_h, self.inp_layer], axis=1)
        #  xW+b part can be calculated altogether
        xW = np.zeros((self.batch, 4*self.hidden_units))
        for inner_idx in range(4):
            xW[:, inner_idx*self.hidden_units:(inner_idx+1)*self.hidden_units] = np.matmul(
                self.state, self.W[inner_idx*self.total_depth:(inner_idx+1)*self.total_depth])
        bias_times_batch = np.tile(self.b, (self.batch, 1))
        self.xW_b = np.add(xW, bias_times_batch)

        # Sigmoid part
        sig_xW_b = self.Sigmoid.forward(self.xW_b[:, :3*self.hidden_units])
        # tangent part
        tanh_xW_b = self.Tanhc.forward(self.xW_b[:, 3*self.hidden_units:])

        self.Hf = sig_xW_b[:, 0*self.hidden_units :1*self.hidden_units]  # forget gate
        self.Hi = sig_xW_b[:, 1*self.hidden_units :2*self.hidden_units]  # input gate
        self.Ho = sig_xW_b[:, 2*self.hidden_units :3*self.hidden_units]  # output gate
        self.Hc = tanh_xW_b                                  # candidate state

        self.c = self.Hf*self.prev_c+self.Hi*self.Hc   # renew state
        self.Tanh_o = self.Tanho.forward(self.c)       # candidate output
        self.h = self.Ho*self.Tanh_o                   # output  (candidate output * sigmoid)
        # return output / next state

        self.local_dL = np.zeros_like(self.state)  # dLoss to be passed down

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
            dc = (dLoss+next_dh)*self.Ho*dTanh_o + next_dc
            dHo = (dLoss+next_dh)*dSigmoid_Ho*self.Tanh_o     #output gate
        else:
            dc = next_dc + next_dh*self.Ho*dTanh_o
            dHo = next_dh*dSigmoid_Ho*self.Tanh_o    #output gate

        dHf = dc*dSigmoid_Hf*self.prev_c       # forget gate
        dHi = dc*dSigmoid_Hi*self.Hc           # input gate
        dHc = dc*dTanh_c*self.Hi           # candidate state

        dbf = dHf
        dbi = dHi
        dbo = dHo
        dbc = dHc
        # concat is slower
        db = np.concatenate((dbf, dbi, dbo, dbc), axis=1)
        self.local_db = np.sum(db, axis=0)

        self.local_dL = np.matmul(dbf, self.W[0*self.total_depth:1*self.total_depth].T)
        self.local_dL += np.matmul(dbi, self.W[1*self.total_depth:2*self.total_depth].T)
        self.local_dL += np.matmul(dbo, self.W[2*self.total_depth:3*self.total_depth].T)
        self.local_dL += np.matmul(dbc, self.W[3*self.total_depth:4*self.total_depth].T)

        self.local_dW[0*self.total_depth:1*self.total_depth] = np.einsum('bi,bj->ij', self.state, dbf)
        self.local_dW[1*self.total_depth:2*self.total_depth] = np.einsum('bi,bj->ij', self.state, dbi)
        self.local_dW[2*self.total_depth:3*self.total_depth] = np.einsum('bi,bj->ij', self.state, dbo)
        self.local_dW[3*self.total_depth:4*self.total_depth] = np.einsum('bi,bj->ij', self.state, dbc)

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
    def __init__(self, fw_Encoder, bw_Encoder):
        """
        1. fw_Encoder : forward encoder (as normal LSTMcell)
        2. bw_Encoder : backward encoder (as normal LSTMcell)
        """
        self.fw_Encoder = fw_Encoder
        self.bw_Encoder = bw_Encoder
        self.fw_hidden_units = fw_Encoder.hidden_units
        self.fw_total_depth = fw_Encoder.total_depth
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
        dW_fw = dW[:4*self.fw_total_depth]
        dW_bw = dW[4*self.fw_total_depth:]
        db_fw = db[:4*self.fw_hidden_units]
        db_bw = db[4*self.fw_hidden_units:]
        self.fw_Encoder.update(dW_fw, db_fw, lr)
        self.bw_Encoder.update(dW_bw, db_bw, lr)
    def get_dWb(self):
        """
        get dW and db from both forward and backward encoder.
        """
        dW_fw, db_fw = self.fw_Encoder.get_dWb()
        dW_bw, db_bw = self.bw_Encoder.get_dWb()
        dW = np.concatenate((dW_fw, dW_bw), axis=0)
        db = np.concatenate((db_fw, db_bw), axis=0)
        return dW, db

    def get_Wb(self):
        """
        get W and b from both forward and backward encoder.
        """
        W_fw, b_fw = self.fw_Encoder.get_Wb()
        W_bw, b_bw = self.bw_Encoder.get_Wb()
        W = np.concatenate((W_fw, W_bw), axis=0)
        b = np.concatenate((b_fw, b_bw), axis=0)
        return W, b


    def rewrite_Wb(self, W, b):
        """
        rewrite W and b in both forward and backward encoder.
        split W into W_fw(forward) and W_bw(backward)
              b into b_fw(forward) and b_bw(backward)
        and deliver these tensor to fw_encoder and bw_encoder
        """
        W_fw = W[:4*self.fw_total_depth]
        W_bw = W[4*self.fw_total_depth:]
        b_fw = b[:4*self.fw_hidden_units]
        b_bw = b[4*self.fw_hidden_units:]

        self.fw_Encoder.rewrite_Wb(W_fw, b_fw)
        self.bw_Encoder.rewrite_Wb(W_bw, b_bw)

@jit
def timestep_softmax_cross_entropy(input_data, target):
    '''
    input:
        1. input_data : shape (timestep x batch x depth)
        2. target     : shape (timestep x batch x depth)
    return prediction(softmax), L, dL
    '''
    after_softmax = np.zeros_like(input_data)
    timesteps = input_data.shape[0]
    batch = input_data.shape[1]
    # softmax
    for timestep in range(timesteps):
        for single_data in range(batch):
            this_row = np.exp(input_data[timestep][single_data])/np.sum(
                np.exp(input_data[timestep][single_data]))
            after_softmax[timestep][single_data] = this_row

    # calculation of L
    small_num = np.zeros_like(target)
    small_num.fill(1e-8)   # prevent log(0)
    L = -np.sum(np.multiply(target, np.log(after_softmax+small_num)))/batch
    # calculation of dL
    All_dLoss = -target + after_softmax
    return after_softmax, L, All_dLoss
def timestep_softmax(input_data):
    '''
    input:
        1. input_data : shape (timestep x batch x depth)
    return prediction(softmax)
    '''
    after_softmax = np.zeros_like(input_data)
    timesteps = input_data.shape[0]
    batch = input_data.shape[1]
    # softmax
    for timestep in range(timesteps):
        for single_data in range(batch):
            this_row = np.exp(input_data[timestep][single_data])/np.sum(
                np.exp(input_data[timestep][single_data]))
            after_softmax[timestep][single_data] = this_row

    return after_softmax


class attention:
    def __init__(self, attention_hidden_units, attention_type='general'):
        """
        np.dot( decoder target layer , np.matmul(W,encoder hidden layer))
        """
        self.attention_type = attention_type
        self.attention_hidden_units = attention_hidden_units
        self.W = np.random.random((attention_hidden_units,
                                   attention_hidden_units))/np.sqrt(attention_hidden_units)
        self.b = np.zeros(1)
        self.db = np.zeros(1)
        self.exist_imaginary_init_decoder_hidden_layer = False
        # be renewed after each training step ( using attribute "update" or "rewrite")
        self.weightized_encoder_hidden_layer = False
        self.context_vector = []
        self.alpha = []
        self.all_decoder_hidden_layer = []
        self.total_decoder_timestep = 0  # to count the amount of decoder time step.
    def forward(self, encoder_hidden_layer, decoder_hidden_layer):

        self.encoder_hidden_layer = encoder_hidden_layer
        self.decoder_hidden_layer = decoder_hidden_layer
        self.all_decoder_hidden_layer.append(decoder_hidden_layer)
        self.encoder_timestep = encoder_hidden_layer.shape[0]
        self.decoder_timestep = decoder_hidden_layer.shape[0]
        self.total_decoder_timestep += decoder_hidden_layer.shape[0]
        self.batch = encoder_hidden_layer.shape[1]
        self.depth = encoder_hidden_layer.shape[2]

        if self.attention_type == 'general':
            #  np.matmul(W,encoder hidden layer),
            #  since input will not change during one forward-backward process
            #  it only needs to be calculated one time.
            #  in einsum, E=encoder timestep, D=decoder timestep, d=depth, b=batch, w=width
            if  self.weightized_encoder_hidden_layer == False:
                self.weightized_encoder_hidden_layer = True
                self.weighted_encoder_hidden_layer = np.einsum('Ebd,dw->Ebw',
                                                               encoder_hidden_layer,
                                                               self.W)

             #  np.dot( decoder target layer , ... ) part
            if  self.weightized_encoder_hidden_layer == True:
                self.this_context_vector = np.zeros((self.decoder_timestep,
                                                     self.batch,
                                                     self.attention_hidden_units))
                this_alpha = np.zeros((self.decoder_timestep, self.batch, self.encoder_timestep))
                for this_decoder_timestep in range(self.decoder_timestep):
                    score = np.einsum('Ebd,bd->bE',
                                      self.weighted_encoder_hidden_layer,
                                      decoder_hidden_layer[this_decoder_timestep])
                    #self.prev_decoder_hidden_layer = decoder_hidden_layer[this_decoder_timestep]
                    # decoder time step /batch / encoder time step
                    score = score - np.max(score)
                    for this_batch in range(self.batch):
                        this_alpha[this_decoder_timestep, this_batch] = \
                        np.exp(score[this_batch, :])/np.sum(np.exp(score[this_batch, :]))

                self.this_context_vector = np.einsum('DbE,Ebd->Dbd',
                                                     this_alpha,
                                                     encoder_hidden_layer)
                self.context_vector.append(self.this_context_vector)
                self.alpha.append(this_alpha)
            #print(np.array(decoder_hidden_layer).shape,np.array(self.context_vector).shape)
            return np.concatenate((decoder_hidden_layer, np.array(self.context_vector)[0]), axis=2)
    def give_me_context_vector(self):
        return np.array(self.context_vector)
    @jit(fastmath=True)
    def backprop(self, dLoss):
        self.alpha = np.array(self.alpha).reshape(self.total_decoder_timestep,
                                                  self.batch,
                                                  self.encoder_timestep)
        self.all_decoder_hidden_layer = np.array(self.all_decoder_hidden_layer).reshape(
            self.total_decoder_timestep, self.batch, self.depth)

        dc = dLoss[:, :, -1*self.attention_hidden_units:]
        self.dW = np.zeros_like(self.W)
        self.ddeInp = np.zeros_like(self.all_decoder_hidden_layer)  # as dLoss
        self.dInp = np.zeros_like(self.encoder_hidden_layer) # to encoder
        for decoder_this_timestep in range(self.total_decoder_timestep):
            for encoder_this_timestep in range(self.encoder_timestep):

                this_alpha = self.alpha[decoder_this_timestep, :, encoder_this_timestep]
                d_this_alpha = this_alpha-this_alpha**2

                dL_a = np.einsum('bd,b->bd', dc[decoder_this_timestep], d_this_alpha)
                dL_a_D = dL_a*self.encoder_hidden_layer[encoder_this_timestep]
                dL_a_D_h = dL_a_D*self.all_decoder_hidden_layer[decoder_this_timestep]

                hW = np.einsum('bd,do->bo',
                               self.encoder_hidden_layer[encoder_this_timestep],
                               self.W)

                self.ddeInp[decoder_this_timestep] += dL_a_D*hW
                self.dInp[encoder_this_timestep] += np.einsum('b,bo->bo',
                                                              this_alpha,
                                                              dc[decoder_this_timestep])
                self.dInp[encoder_this_timestep] += np.einsum('bd,do->bo',
                                                              dL_a_D_h,
                                                              self.W.T)

        #return self.ddeInp, self.dInp
        return dLoss[:, :, :-1*self.attention_hidden_units]+self.ddeInp, self.dInp

    def update(self, dW, db, lr):
        """
        update dW and db.
        """
        self.W = self.W - lr*dW
        self.end_of_this_step()

    def get_dWb(self):
        """
        get dW and db.
        """
        return self.dW, self.db

    def get_Wb(self):
        """
        get W and b.
        """
        return self.W, self.b

    def get_alpha(self):
        """
        get alpha in shape (Decoder_timestep, batch, encoder timestep)
        """
        return np.array(self.alpha)

    def rewrite_Wb(self, W, b):
        """
        rewrite W and b.
        """
        self.W = W
        self.end_of_this_step()

    def end_of_this_step(self):
        """
        to clean up the memory for this step.
        """
        self.weightized_encoder_hidden_layer = False
        self.context_vector = []
        self.alpha = []
        self.total_decoder_timestep = 0
        self.all_decoder_hidden_layer = []
