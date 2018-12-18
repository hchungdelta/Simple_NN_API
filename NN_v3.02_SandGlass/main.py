"""
Title : SandGlass Conv2Conv model, training code
Description :
Sanity check for attention mechanism,
whether it can reverse the input as output.
For example,
input = [1,3,5,2,0]
target = [0,2,5,3,1]
Author: Hao-Chien, Hung
Date: 15/12/2018
"""
import time
import numpy as np
import ML
from random_gene import Random_input_generator, input_helper, ThreeD_onehot

# mpi4py
comm = ML.EndeModel.comm
rank = ML.EndeModel.rank
size = ML.EndeModel.size

# set print options (optional)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# training data information
batch = 100
length = 24
vocab_upper_limit = 60
input_max_equal_output = False
# save path
savepath = 'data/1218.pickle'

# encoder 1st conv layer
conv1D_0 = ML.Layer.CNN2CNN_layer.conv1D((1, 1, 1, vocab_upper_limit, 60), residual=False)
# encoder 2nd conv layer (multiple conv)
conv1D_1a = ML.Layer.CNN2CNN_layer.conv1D((1, 1, 4, 40, 60), residual=False)
conv1D_1b = ML.Layer.CNN2CNN_layer.conv1D((1, 1, 1, 40, 60), residual=False)
conv1Ds_1 = ML.Layer.CNN2CNN_layer.conv1Ds([conv1D_1a, conv1D_1b])
# encoder 3rd conv layer
conv1D_2 = ML.Layer.CNN2CNN_layer.conv1D((2, 1, 3, 40, 60), residual=False)

# encoder 4th conv layer (reduce conv)
ReduceConv1 = ML.Layer.CNN2CNN_layer.ReduceConv((4, 40, 60))

# center Fully Connected layer
FCLlayer1 = ML.Layer.FCL_layer.xW_b((40*6, 60*6))

# decoder 1st conv layer (expand conv)
ExpandConv1 = ML.Layer.CNN2CNN_layer.ExpandConv((4, 80, 60))

# decoder 2nd conv layer
conv1D_rev_1 = ML.Layer.CNN2CNN_layer.conv1D_rev((1, 2, 3, 80, 60), residual=False)
# decoder 3rd conv layer (multiple conv)
conv1D_rev_2a = ML.Layer.CNN2CNN_layer.conv1D_rev((1, 1, 4, 80, 60), residual=False)
conv1D_rev_2b = ML.Layer.CNN2CNN_layer.conv1D_rev((1, 1, 1, 80, 60), residual=False)
conv1Ds_rev_2 = ML.Layer.CNN2CNN_layer.conv1Ds_rev([conv1D_rev_2a, conv1D_rev_2b])
# decoder 4th conv layer
conv1D_rev_3 = ML.Layer.CNN2CNN_layer.conv1D_rev((1, 1, 1, 80, vocab_upper_limit), residual=False)

attn0 = ML.Layer.Attention.DotAttn()
attn_helper_0 = ML.Layer.Attention.Attn_helper(attn0, 40)
attn1 = ML.Layer.Attention.DotAttn()
attn_helper_1 = ML.Layer.Attention.Attn_helper(attn1, 40)
attn2 = ML.Layer.Attention.DotAttn()
attn_helper_2 = ML.Layer.Attention.Attn_helper(attn2, 40)
attn3 = ML.Layer.Attention.DotAttn()
attn_helper_3 = ML.Layer.Attention.Attn_helper(attn3, 40)

# Batch Normalization
BN0 = ML.Layer.Normalization.BatchNorm_4d((40), eps=1e-6, mode='train')
BN1 = ML.Layer.Normalization.BatchNorm_4d((40), eps=1e-6, mode='train')
BN2 = ML.Layer.Normalization.BatchNorm_4d((40), eps=1e-6, mode='train')
BN3 = ML.Layer.Normalization.BatchNorm_4d((40), eps=1e-6, mode='train')
BN4 = ML.Layer.Normalization.BatchNorm_4d((80), eps=1e-6, mode='train')
BN5 = ML.Layer.Normalization.BatchNorm_4d((80), eps=1e-6, mode='train')
BN6 = ML.Layer.Normalization.BatchNorm_4d((80), eps=1e-6, mode='train')
BN7 = ML.Layer.Normalization.BatchNorm_4d((80), eps=1e-6, mode='train')
DROP_2d = ML.Layer.FCL_layer.dropout_2d(keep_prob=0.8)
# others
Flat = ML.Layer.CNN2CNN_layer.flatten()
RevFlat = ML.Layer.CNN2CNN_layer.rever_flatten(timestep=6, depth=60)
Expand_dims = ML.Layer.CNN2CNN_layer.expand_dims(0)
Squeeze_dims = ML.Layer.CNN2CNN_layer.squeeze(0)

# build up architecture
EndeModel = ML.EndeModel.Model(lr=0.0007, mode='adam', clipping=True, clip_value=1.00)

#### encoder part (stimulus)
EndeModel.add(Expand_dims)
EndeModel.add(conv1D_0)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(attn_helper_0)
EndeModel.add(BN0)

EndeModel.add(conv1Ds_1)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(attn_helper_1)
EndeModel.add(BN1)

EndeModel.add(conv1D_2)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(attn_helper_2)
EndeModel.add(BN2)

EndeModel.add(ReduceConv1)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(attn_helper_3)
EndeModel.add(BN3)

#### center part (brain processing)
EndeModel.add(Flat)

EndeModel.add(FCLlayer1)
EndeModel.add(RevFlat)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(attn_helper_3)
EndeModel.add(BN4)

#### decoder part (response)
EndeModel.add(ExpandConv1)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(attn_helper_2)
EndeModel.add(BN5)

EndeModel.add(conv1D_rev_1)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(attn_helper_1)
EndeModel.add(BN6)

EndeModel.add(conv1Ds_rev_2)
EndeModel.add(ML.Layer.Acti_layer.Tanh())
EndeModel.add(attn_helper_0)
EndeModel.add(BN7)

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

for step in range(15000):
    random_generator = Random_input_generator(batchsize=batch,
                                              vocab_lower_limit=2,
                                              vocab_upper_limit=vocab_upper_limit,
                                              length_lower_limit=3,
                                              length_upper_limit=length-1)
    batchinput, _, batchtarget = input_helper(random_generator, length)
    batchinput = ThreeD_onehot(batchinput, vocab_upper_limit)
    batchtarget = ThreeD_onehot(batchtarget, vocab_upper_limit)
    target = batchtarget[1:][::-1]
    feed_input = batchinput


    # training
    output = EndeModel.Forward(feed_input)
    pred, L, dLoss = ML.NN.Loss.timestep_softmax_cross_entropy(output, target)
    dLoss = EndeModel.Backprop(dLoss)

    tot_L += L/display
    if rank == 0 and step%display == 0:
        print("Loss: {:.3f} ".format(tot_L))
        tot_L = 0
        TARGET = np.argmax(target, axis=2).transpose(1, 0)
        PRED = np.argmax(pred, axis=2).transpose(1, 0)
        for number in range(1):
            print("TARGET:", TARGET[number])
            print("PRED:", PRED[number])
            print("INP", TARGET[number][::-1])
        print("cost : {:.3f} seconds".format(time.time()-display_time))
        EndeModel.Save(savepath)
        display_time = time.time()
