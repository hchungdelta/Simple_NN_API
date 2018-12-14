"""
Title : SandGlass Conv2Conv model, interactive infer code

Author: Hao-Chien, Hung
Date: 15/12/2018
"""
import sys
import numpy as np
import ML
from data_importer import input_dict, input_training_data
from data_importer import encode, decode, ThreeD_onehot

# mpi4py
comm = ML.EndeModel.comm
rank = ML.EndeModel.rank
size = ML.EndeModel.size

# set print options (optional)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# input training data/dict
dict_path = "trainingdata/dict_test30.json"
training_data_path = "trainingdata/training_30data.json"
dicts, reversed_dicts = input_dict(dict_path)
inputdata, targetdata = input_training_data(training_data_path)



# training data information
data_input_depth = 32
batch = 1
length = 32
vocab_upper_limit = len(dicts)
input_max_equal_output = False

# save path
savepath = 'data/1208.pickle'

# encoder 1st conv layer (multiple conv)
# conv info in shape (input_filters,output_filters,kernel_amount,input_depth,output_depth)
conv1D_1a = ML.Layer.CNN2CNN_layer.conv1D((1, 1, 4, vocab_upper_limit, 50), residual=False)
conv1D_1b = ML.Layer.CNN2CNN_layer.conv1D((1, 1, 3, vocab_upper_limit, 50), residual=False)
conv1Ds_1 = ML.Layer.CNN2CNN_layer.conv1Ds([conv1D_1a, conv1D_1b])

# encoder 2nd conv layer
conv1D_2 = ML.Layer.CNN2CNN_layer.conv1D((2, 1, 3, 50, 50), residual=False)

# encoder 3rd conv layer
ReduceConv1 = ML.Layer.CNN2CNN_layer.ReduceConv((4, 50, 50))
#ReduceAttn1 = ML.Layer.CNN2CNN_layer.ReduceAttn(4)

# center Fully Connected layer
FCLlayer1 = ML.Layer.FCL_layer.xW_b((50*8, 50*8))

# decoder 1st conv layer
ExpandConv1 = ML.Layer.CNN2CNN_layer.ExpandConv((4, 50, 50))

# decoder 2nd conv layer
conv1D_rev_1 = ML.Layer.CNN2CNN_layer.conv1D_rev((1, 2, 3, 50, 50), residual=False)

# decoder 3rd conv layer
conv1D_rev_2a = ML.Layer.CNN2CNN_layer.conv1D_rev((1, 1, 4, 50, 50), residual=False)
conv1D_rev_2b = ML.Layer.CNN2CNN_layer.conv1D_rev((1, 1, 3, 50, 50), residual=False)
conv1Ds_rev_2 = ML.Layer.CNN2CNN_layer.conv1Ds_rev([conv1D_rev_2a, conv1D_rev_2b])

conv1D_rev_3 = ML.Layer.CNN2CNN_layer.conv1D_rev((1, 1, 1, 50, vocab_upper_limit), residual=False)

# Batch Normalization
BN1 = ML.Layer.Normalization.BatchNorm_4d((50), eps=1e-5, mode='infer')
BN2 = ML.Layer.Normalization.BatchNorm_4d((50), eps=1e-5, mode='infer')
BN3 = ML.Layer.Normalization.BatchNorm_4d((50), eps=1e-5, mode='infer')
BNFCL = ML.Layer.Normalization.BatchNorm_2d((400), eps=1e-5, mode='infer')
BN4 = ML.Layer.Normalization.BatchNorm_4d((50), eps=1e-5, mode='infer')
BN5 = ML.Layer.Normalization.BatchNorm_4d((50), eps=1e-5, mode='infer')
BN6 = ML.Layer.Normalization.BatchNorm_4d((50), eps=1e-5, mode='infer')

DROP_2d = ML.Layer.FCL_layer.dropout_2d(keep_prob=1.0)
DROP_4d_1 = ML.Layer.FCL_layer.dropout_4d(keep_prob=1.0)
DROP_4d_2 = ML.Layer.FCL_layer.dropout_4d(keep_prob=1.0)
DROP_4d_3 = ML.Layer.FCL_layer.dropout_4d(keep_prob=1.0)
DROP_4d_4 = ML.Layer.FCL_layer.dropout_4d(keep_prob=1.0)
DROP_4d_5 = ML.Layer.FCL_layer.dropout_4d(keep_prob=1.0)
DROP_4d_6 = ML.Layer.FCL_layer.dropout_4d(keep_prob=1.0)

# others
Flat = ML.Layer.CNN2CNN_layer.flatten()
RevFlat = ML.Layer.CNN2CNN_layer.rever_flatten(timestep=8, depth=50)
Expand_dims = ML.Layer.CNN2CNN_layer.expand_dims(0)
Squeeze_dims = ML.Layer.CNN2CNN_layer.squeeze(0)

# build up architecture
EndeModel = ML.EndeModel.Model(lr=0.0010, mode='adam', clipping=True, clip_value=0.20)

#encoder part (stimulus)
EndeModel.add(Expand_dims)

EndeModel.add(conv1Ds_1)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(BN1)
EndeModel.add(DROP_4d_1)

EndeModel.add(conv1D_2)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(BN2)
EndeModel.add(DROP_4d_2)

EndeModel.add(ReduceConv1) #EndeModel.add(ReduceAttn1)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(BN3)
EndeModel.add(DROP_4d_3)

# center part (brain processing)
EndeModel.add(Flat)
EndeModel.add(FCLlayer1)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(BNFCL)
EndeModel.add(DROP_2d)
EndeModel.add(RevFlat)

# decoder part (response)
EndeModel.add(ExpandConv1)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(BN4)
EndeModel.add(DROP_4d_4)

EndeModel.add(conv1D_rev_1)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(BN5)
EndeModel.add(DROP_4d_5)

EndeModel.add(conv1Ds_rev_2)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(BN6)
EndeModel.add(DROP_4d_6)
EndeModel.add(conv1D_rev_3)

EndeModel.add(Squeeze_dims)

# show further information
if rank == 0:
    print(EndeModel.show_detail())


EndeModel.Restore(savepath)

while True:
    keyin = input("input :")
    if keyin == "exit()":
        sys.exit()
    encoded_keyin = encode(keyin, dicts)
    encoded_keyin.append(1)
    if len(encoded_keyin) < data_input_depth:
        encoded_keyin.extend([0]*(data_input_depth-len(encoded_keyin)))
    encoded_keyin = np.array([encoded_keyin])

    feed_input = ThreeD_onehot(encoded_keyin, vocab_upper_limit).transpose(1, 0, 2)
    output = EndeModel.Forward(feed_input)
    pred = ML.NN.Loss.softmax(output)

    ENINP = np.argmax(feed_input, axis=2).transpose(1, 0)
    PRED = np.argmax(pred, axis=2).transpose(1, 0)
    print("input  : ", decode(ENINP[0].tolist(), reversed_dicts))
    print("pred   : ", decode(PRED[0].tolist(), reversed_dicts))
