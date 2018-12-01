#!/usr/bin/env python3
# -*- coding: utf_8 -*-

'''
mpi4py:
comm = parallel computing commander  /None if MPI not exist
rank = index of this processor       /0    if MPI not exist
size = amount of processors          /1    if MPI not exist
'''
import time
import numpy as np
import ML
from data_importer import input_helper, decode
from data_importer import Input_training_data, Input_dictionary_and_Embedding

# mpi4py
COMM = ML.EndeModel.comm
RANK = ML.EndeModel.rank
SIZE = ML.EndeModel.size

# start / end token
GO_TOKEN = 40849
END_TOKEN = 40850

# input data
dictionary, reverse_dictionary, Embedding = Input_dictionary_and_Embedding()
encoder_inputs_data, decoder_targets_data, amount_of_sentence = Input_training_data()
amount_of_vocab = len(dictionary)
vocab_vector_size = len(Embedding[0])

# word 2 vector layer. (using genism)
_embeddings = [float(i) for vector in Embedding  for i in vector]
embeddings = np.reshape(_embeddings, [amount_of_vocab, vocab_vector_size])

#save path for trainable variables
savepath = 'data/1130.pickle'

#parameter
batch = 4
hidden_units = 600
input_depth = vocab_vector_size 
deinput_depth = vocab_vector_size
length = 21  # max time step
teacher_force = True
display_epoch = 50

# timestep, batch, input_units, output_units
encoder_info1 = [length, batch, input_depth, hidden_units]
decoder_info1 = [length, batch, deinput_depth, 2*hidden_units]

# Build up neural network architecture for all processors
# first encoder layer
fw_Encoder1 = ML.Layer.LSTM_layer.LSTMcell(encoder_info1, output_form="None")
bw_Encoder1 = ML.Layer.LSTM_layer.LSTMcell(encoder_info1, output_form="None")
Bi_Encoder1 = ML.Layer.LSTM_layer.BiLSTM(fw_Encoder1, bw_Encoder1)

# first decoder layer
Decoder1 = ML.Layer.LSTM_layer.LSTMcell(decoder_info1, output_form="All")

# for 'infer' training approach
Decoders = ML.Layer.LSTM_layer.InferLSTMcell([Decoder1])

linearlayer = ML.Layer.FCL_layer.timestep_xW_b((2*hidden_units, amount_of_vocab))
Embedding_matrix = ML.Layer.FCL_layer.Embedding(embeddings)

# architecture
EndeModel = ML.EndeModel.Model(lr=0.0012)
EndeModel.add(Bi_Encoder1)
EndeModel.add(Decoder1)
EndeModel.add(linearlayer)

# initializer
if EndeModel.comm != None :
    EndeModel.Bcast_Wb(initial=True)

EndeModel.Restore('data/1124.pickle')

displaytime = time.time()
total_L = 0
offset = 0
Finished_loop = 0
for a in range(int(100001/SIZE)):
    # input
    if offset+(RANK+1)*batch > len(encoder_inputs_data):
        offset = 0
        Finished_loop += 1
    batchinput, batch_decode_input, batchtarget = input_helper(encoder_inputs_data,
                                                               decoder_targets_data,
                                                               offset+RANK*batch,
                                                               offset+(RANK+1)*batch,
                                                               amount_of_vocab,
                                                               GO_TOKEN,
                                                               END_TOKEN)
    offset += batch*SIZE
    batchinput_maxlength = 21
    batchtarget_maxlength = 21
    # separate to N processors

    en_max_length = batchinput_maxlength
    de_max_length = batchtarget_maxlength+1
    target = batchtarget
    # forward propagation

    # pass w2v before entering LSTM
    wordvector = Embedding_matrix.select(batchinput)

    # return (h_list, c_list, final_h_state, final_c_state)
    _, _, en1_final_h, en1_final_c = Bi_Encoder1.forward(wordvector,
                                                         None,
                                                         None,
                                                         cutoff_length=en_max_length)



    if teacher_force == True:
        decode_input_wordvector = Embedding_matrix.select(batch_decode_input)
        de1_h_list, _, _, _ = Decoder1.forward(decode_input_wordvector,
                                               en1_final_h,
                                               en1_final_c,
                                               cutoff_length=de_max_length)
        output = linearlayer.forward(de1_h_list)
        pred, L, dLoss_list = ML.Layer.LSTM_layer.timestep_softmax_cross_entropy(output, target)

    else:
        encoder_states = [[en1_final_h], [en1_final_c]]
        pred, L, dLoss_list = Decoders.forward(encoder_states,
                                               linearlayer,
                                               Embedding_matrix,
                                               target,
                                               cutoff_length=de_max_length)

    # back prop ( FCL part)
    dLoss_list = linearlayer.backprop(dLoss_list, random=True)

    # back prop (LSTM part)
    _, de1_dh, de1_dc = Decoder1.backprop(dLoss_list,
                                          None,
                                          None,
                                          cutoff_length=de_max_length)
    Bi_Encoder1.backprop(None,
                         de1_dh,
                         de1_dc,
                         cutoff_length=en_max_length)

    EndeModel.Update_all(mode="adam")

    total_L += L/display_epoch
    if a%display_epoch == 0 and RANK == 0:
        print("Finished {}/{} , finished loop {}".format(offset,
                                                         amount_of_sentence,
                                                         Finished_loop))
        print("Loss {:.5f}".format(total_L))
        PRED = np.argmax(pred, axis=2)
        INPUT = np.argmax(batchinput, axis=2)
        DEINPUT = np.argmax(batch_decode_input, axis=2)
        PRED = PRED.transpose(1, 0)
        INPUT = INPUT.transpose(1, 0)
        DEINPUT = DEINPUT.transpose(1, 0)
        for number in range(3):
            print(number)
            print(INPUT[number])
            print(PRED[number])
            print("INPUT  : ", decode(INPUT[number], reverse_dictionary))
            print("PRED   : ", decode(PRED[number], reverse_dictionary))
            print("DEINP  : ", decode(DEINPUT[number], reverse_dictionary))

        total_L = 0
        print("time cost {:.2f} seconds ,for one display epoch".format(
            time.time()-displaytime))
        displaytime = time.time()
        EndeModel.Save(savepath)
