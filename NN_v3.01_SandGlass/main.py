"""
Title : SandGlass Conv2Conv model, training code

Author: Hao-Chien, Hung
Date: 15/12/2018
"""
import time
import numpy as np
import ML
from data_importer import input_dict, input_training_data, input_Embedding
from data_importer import batch_helper, input_helper, decode

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
batch = 100
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
BN1 = ML.Layer.Normalization.BatchNorm_4d((50), eps=1e-5, mode='train')
BN2 = ML.Layer.Normalization.BatchNorm_4d((50), eps=1e-5, mode='train')
BN3 = ML.Layer.Normalization.BatchNorm_4d((50), eps=1e-5, mode='train')
BNFCL = ML.Layer.Normalization.BatchNorm_2d((400), eps=1e-5, mode='train')
BN4 = ML.Layer.Normalization.BatchNorm_4d((50), eps=1e-5, mode='train')
BN5 = ML.Layer.Normalization.BatchNorm_4d((50), eps=1e-5, mode='train')
BN6 = ML.Layer.Normalization.BatchNorm_4d((50), eps=1e-5, mode='train')

DROP_2d = ML.Layer.FCL_layer.dropout_2d(keep_prob=0.8)
DROP_4d_1 = ML.Layer.FCL_layer.dropout_4d(keep_prob=0.8)
DROP_4d_2 = ML.Layer.FCL_layer.dropout_4d(keep_prob=0.8)
DROP_4d_3 = ML.Layer.FCL_layer.dropout_4d(keep_prob=0.8)
DROP_4d_4 = ML.Layer.FCL_layer.dropout_4d(keep_prob=0.8)
DROP_4d_5 = ML.Layer.FCL_layer.dropout_4d(keep_prob=0.8)
DROP_4d_6 = ML.Layer.FCL_layer.dropout_4d(keep_prob=0.8)

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

# initializer
if EndeModel.comm != None:
    EndeModel.Bcast_Wb(initial=True)

#EndeModel.Restore(savepath)

tot_L = 0
display = 30

# record the cost of time
display_time = time.time()

for step in range(5000):
    if rank == 0:
        batchinput, batch_decode_input, batchtarget = batch_helper(inputdata,
                                                                   targetdata,
                                                                   batch*size,
                                                                   length,
                                                                   mode='nocut')
        bcast_data = {"bathinput":batchinput,
                      "batch_decode_input":batch_decode_input,
                      "batchtarget":batchtarget}
    else:
        bcast_data = None
    if EndeModel.comm != None:
        bcast_data = comm.bcast(bcast_data, root=0)
        batchinput = bcast_data["bathinput"]
        batch_decode_input = bcast_data["batch_decode_input"]
        batchtarget = bcast_data["batchtarget"]


    # separate to N processors
    batchinput = batchinput[batch*rank:batch*(rank+1)]
    batchtarget = batchtarget[batch*rank:batch*(rank+1)]
    batch_decode_input = batch_decode_input[batch*rank:batch*(rank+1)]
    batchinput = input_helper(batchinput, vocab_upper_limit)
    batchtarget = input_helper(batchtarget, vocab_upper_limit)
    batch_decode_input = input_helper(batch_decode_input, vocab_upper_limit)
    target = batchtarget
    feed_input = batchinput

    output, forward_dict = EndeModel.Forward(feed_input, show=True, show_type='absmean')
    pred, L, dLoss = ML.NN.Loss.timestep_softmax_cross_entropy(output, target)
    dL, backprop_dict = EndeModel.Backprop(dLoss, show=True, show_type='absmean')

    tot_L += L/display
    if rank == 0 and step%display == 0:
        print("Loss: {:.3f} ".format(tot_L))
        print("forward", forward_dict)
        print("backward", backprop_dict)
        tot_L = 0
        ENINP = np.argmax(feed_input, axis=2).transpose(1, 0)
        TARGET = np.argmax(target, axis=2).transpose(1, 0)
        PRED = np.argmax(pred, axis=2).transpose(1, 0)
        for number in range(1):
            print("sample ", number)
            print("input  : ", decode(ENINP[number].tolist(), reversed_dicts))
            print("target : ", decode(TARGET[number].tolist(), reversed_dicts))
            print("pred   : ", decode(PRED[number].tolist(), reversed_dicts))

        print("cost : {:.3f} seconds".format(time.time()-display_time))
        EndeModel.Save(savepath)
        display_time = time.time()
