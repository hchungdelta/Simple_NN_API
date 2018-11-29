#!/usr/bin/env python3
# -*- coding: utf_8 -*-

'''
Title : Encoder-decoder neural network (LSTM)
Description : This code is for test.

exit() = SystemExit

Author : Hao Chien, Hung
date : 28/11/2018

mpi4py:
comm = parallel computing commander  /None if MPI not exist
rank = index of this processor       /0    if MPI not exist
size = amount of processors          /1    if MPI not exist
'''
import sys
import numpy as np
import ML
from data_importer import decode, encode, spec_vocab_searcher, ThreeD_onehot
from data_importer import Input_dictionary_and_Embedding

def testmode():
    """
    for test, the parameters must be the same as those in the training mode.
    """
    # input data
    dictionary, reversed_dictionary, Embedding_matrix = Input_dictionary_and_Embedding()
    amount_of_vocab = len(dictionary)
    vocab_vector_size = len(Embedding_matrix[0])

    # word 2 vec layer.
    _embeddings = [float(i)  for vector in Embedding_matrix for i in vector]
    embeddings = np.reshape(_embeddings, [amount_of_vocab, vocab_vector_size])  
    # parameter
    batch = 1
    hidden_units = 600
    input_depth = vocab_vector_size
    deinput_depth = vocab_vector_size
    length = 21  # max time step

    # timestep, batch, input_units, output_units
    encoder_info1 = [length, batch, input_depth, hidden_units]
    decoder_info1 = [length, batch, deinput_depth, 2*hidden_units]

    # Build up neural network architecture for all processors
    #encoder layer
    fw_Encoder1 = ML.Layer.LSTM_layer.LSTMcell(encoder_info1, output_form="None")
    bw_Encoder1 = ML.Layer.LSTM_layer.LSTMcell(encoder_info1, output_form="None")
    Bi_Encoder1 = ML.Layer.LSTM_layer.BiLSTM(fw_Encoder1, bw_Encoder1)

    #decoder layer
    Decoder1 = ML.Layer.LSTM_layer.LSTMcell(decoder_info1, output_form="All")

    # for 'infer' training approach
    Decoders = ML.Layer.LSTM_layer.InferLSTMcell([Decoder1])

    # Fully connnected layer
    linearlayer = ML.Layer.FCL_layer.timestep_xW_b((2*hidden_units, amount_of_vocab))
    Embedding_matrix = ML.Layer.FCL_layer.Embedding(embeddings)

    # architecture
    EndeModel = ML.EndeModel.Model(lr=0.0012)
    EndeModel.add(Bi_Encoder1)
    EndeModel.add(Decoder1)
    EndeModel.add(linearlayer)
    # restore data
    EndeModel.Restore('data/1124.pickle')


    while True:
        keyin = input("input :")
        if keyin == "exit()":
            sys.exit()
        keyin = keyin+'<EOS>'
        # to find "keywords" in the input sentence
        keyword_keyin = spec_vocab_searcher(keyin, dictionary)
        # encode the sentence
        encoded_keyin = encode(keyword_keyin, dictionary)
        # reshape the sentence
        encoder_input = np.array(encoded_keyin).reshape(1, -1)
        # reshape the sentence
        encoder_input = ThreeD_onehot(encoder_input, amount_of_vocab).transpose(1, 0, 2)

        #target = encoder_input
        en_max_length = len(encoded_keyin)
        de_max_length = 21

        # pass w2v before entering LSTM
        wordvector = Embedding_matrix.select(encoder_input)
        # for decoder input
        _, _, en1_final_h, en1_final_c = Bi_Encoder1.forward(wordvector,
                                                             None,
                                                             None,
                                                             cutoff_length=en_max_length)

        encoder_states = [[en1_final_h], [en1_final_c]]
        pred = Decoders.just_forward(encoder_states,
                                     linearlayer,
                                     Embedding_matrix,
                                     cutoff_length=de_max_length)
        # decode part
        INPUT = np.argmax(encoder_input, axis=2).transpose(1, 0)
        PRED = np.argmax(pred, axis=2).transpose(1, 0)

        print("INPUT  : ", decode(INPUT[0], reversed_dictionary))
        print("PRED   : ", decode(PRED[0], reversed_dictionary))

if __name__ == "__main__":
    testmode()
