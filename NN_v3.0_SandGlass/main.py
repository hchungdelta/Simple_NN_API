"""
Title : SandGlass Conv2Conv model, training code

Author: Hao-Chien, Hung
Date: 05/12/2018
"""
import time
import numpy as np
import ML
from data_importer import  decode, batch_helper, input_helper, input_dict, input_training_data

# mpi4py
comm = ML.EndeModel.comm
rank = ML.EndeModel.rank
size = ML.EndeModel.size

# set print options (optional)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# input training data
dict_path = "trainingdata/dict_test30.json"
training_data_path = "trainingdata/training_30data.json"
dicts, reversed_dicts = input_dict(dict_path)
inputdata, targetdata = input_training_data(training_data_path)

# training data information
data_input_depth = len(dicts)
data_input_batch = 10
data_input_length = 32
input_max_equal_output = False

# save path
savepath = 'data/1205.pickle'

# encoder 1st conv layer (multiple conv)
# conv info in shape (input_filters,output_filters,kernel_amount,input_depth,output_depth)
conv1D_1a = ML.Layer.CNN2CNN_layer.conv1D((1, 1, 4, data_input_depth, 40), residual=False)
conv1D_1b = ML.Layer.CNN2CNN_layer.conv1D((1, 1, 3, data_input_depth, 40), residual=False)
conv1Ds_1 = ML.Layer.CNN2CNN_layer.conv1Ds([conv1D_1a, conv1D_1b])

# encoder 2nd conv layer
conv1D_2 = ML.Layer.CNN2CNN_layer.conv1D((2, 1, 3, 40, 40), residual=False)

# encoder 3rd conv layer
ReduceConv1 = ML.Layer.CNN2CNN_layer.ReduceConv((4, 40, 40))
#ReduceAttn1 = ML.Layer.CNN2CNN_layer.ReduceAttn(4)

# center Fully Connected layer
FCLlayer1 = ML.Layer.FCL_layer.xW_b((40*8, 40*8))


# decoder 1st conv layer
ExpandConv1 = ML.Layer.CNN2CNN_layer.ExpandConv((4, 40, 40))

# decoder 2nd conv layer
conv1D_rev_1 = ML.Layer.CNN2CNN_layer.conv1D_rev((1, 2, 3, 40, 40), residual=False)

# decoder 3rd conv layer
conv1D_rev_2a = ML.Layer.CNN2CNN_layer.conv1D((1, 1, 4, 40, data_input_depth), residual=False)
conv1D_rev_2b = ML.Layer.CNN2CNN_layer.conv1D((1, 1, 3, 40, data_input_depth), residual=False)
conv1Ds_rev_2 = ML.Layer.CNN2CNN_layer.conv1Ds_rev([conv1D_rev_2a, conv1D_rev_2b])

# Batch Normalization
BN1 = ML.Layer.CNN2CNN_layer.BatchNorm(mode='train')
BN2 = ML.Layer.CNN2CNN_layer.BatchNorm(mode='train')
BN3 = ML.Layer.CNN2CNN_layer.BatchNorm(mode='train')

# others
Flat = ML.Layer.CNN2CNN_layer.flatten()
RevFlat = ML.Layer.CNN2CNN_layer.rever_flatten((1, 8, data_input_batch, -1))
Expand_dims = ML.Layer.CNN2CNN_layer.expand_dims(0)
Squeeze_dims = ML.Layer.CNN2CNN_layer.squeeze(0)

# build up architecture
EndeModel = ML.EndeModel.Model(lr=0.002, mode='adam', clipping=True, clip_value=1)

#encoder part (stimulus)
EndeModel.add(Expand_dims)
EndeModel.add(conv1Ds_1)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(BN1)
EndeModel.add(conv1D_2)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(BN2)
EndeModel.add(ReduceConv1) #EndeModel.add(ReduceAttn1)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(BN3)

# center part (brain processing)
EndeModel.add(Flat)
EndeModel.add(FCLlayer1)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(RevFlat)

# decoder part (response)
EndeModel.add(ExpandConv1)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(conv1D_rev_1)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(conv1Ds_rev_2)
EndeModel.add(Squeeze_dims)

# show further information
if rank == 0:
    print(EndeModel.show_detail())

# initializer
if EndeModel.comm != None:
    EndeModel.Bcast_Wb(initial=True)

#EndeModel.Restore(savepath)

tot_L = 0
display = 100

# record the cost of time
display_time = time.time()

for step in range(5000):
    if rank == 0:
        batchinput, batch_decode_input, batchtarget = batch_helper(inputdata,
                                                                   targetdata,
                                                                   data_input_batch,
                                                                   data_input_length,
                                                                   mode="nocut")
        bcast_data = {"bathinput":batchinput,
                      "batch_decode_input":batch_decode_input,
                      "batchtarget":batchtarget}
    else:
        bcast_data = None

    if comm != None:
        bcast_data = comm.bcast(bcast_data, root=0)
        batchinput = bcast_data["bathinput"]
        batch_decode_input = bcast_data["batch_decode_input"]
        batchtarget = bcast_data["batchtarget"]

    # separate to N processors
    batchinput = batchinput[data_input_batch*rank:data_input_batch*(rank+1)]
    batchtarget = batchtarget[data_input_batch*rank:data_input_batch*(rank+1)]
    batchinput = input_helper(batchinput, data_input_depth)
    batchtarget = input_helper(batchtarget, data_input_depth)
    target = batchtarget

    # training
    output = EndeModel.Forward(batchinput)
    pred, L, dLoss = ML.NN.Loss.timestep_softmax_cross_entropy(output, target)
    EndeModel.Backprop(dLoss)

    tot_L += L/display
    if rank == 0 and step%display == 0:
        print("Loss: {:.3f} ".format(tot_L))
        tot_L = 0
        ENINP = np.argmax(batchinput, axis=2).transpose(1, 0)
        TARGET = np.argmax(target, axis=2).transpose(1, 0)
        PRED = np.argmax(pred, axis=2).transpose(1, 0)
        for number in range(3):
            print("sample ", number)
            print("input  : ", decode(ENINP[number].tolist(), reversed_dicts))
            print("target : ", decode(TARGET[number].tolist(), reversed_dicts))
            print("pred   : ", decode(PRED[number].tolist(), reversed_dicts))
        print("cost : {:.3f} seconds".format(time.time()-display_time))
        #ndeModel.Save(savepath)
        display_time = time.time()
