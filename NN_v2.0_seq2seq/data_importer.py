#!/usr/bin/env python3
# -*- coding: utf_8 -*-

'''
Input training data, dictionary and some helpful functions.
'''
import json
import numpy as np

def decode(encoded_sentence, reverse_dictionary):
    '''
    [13, 23, 15, 1, 0]  --> I am fine<EOS>
    '''
    decoded_sentence = ''
    for index in  encoded_sentence:
        if reverse_dictionary[index] != ' ':
            decoded_sentence += str(reverse_dictionary[index])
    return decoded_sentence

def encode(raw_sentence, dictionary):
    '''
    I am fine<EOS> --> [13, 23, 15, 1, 0]
    '''
    encoded_sentence = []
    for word in raw_sentence:
        try:
            encoded_sentence.append(dictionary[word])# if in dictionary
        except:
            encoded_sentence.append(dictionary["unknown"])# if not in dictionary
    return encoded_sentence

def spec_vocab_searcher(_string, dictionary):
    """
    separate the sentence into unique words (defined by dictionary) 
    """
    if type(_string) == str:
        list_input_phase = list(_string)
    pharse_include_spec_vocab = []
    char_num = 0
    up_limit = 7
    while char_num < (len(list_input_phase)):
        match = False
        sub_dict = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}

        if (len(list_input_phase)-char_num) < 7:
            up_limit = len(list_input_phase) - char_num  +1
        try:
            for vocab in dictionary.keys():
                if vocab.startswith(list_input_phase[char_num]) and match == False:
                    sub_dict[len(vocab)].append(vocab)
            for order in reversed(range(1, up_limit)):
                if match == False:
                    txt = ''
                    for length in range(order):
                        txt += list_input_phase[char_num+length]
                    if txt in sub_dict[order]:
                        pharse_include_spec_vocab += [txt]
                        match = True
                        char_num += (len(txt))
        except:
            pass
        if match == False:
            pharse_include_spec_vocab += list_input_phase[char_num]
            char_num += 1
    return pharse_include_spec_vocab

def flatten(_input, split=1):       # used to input data.
    if split == 0:
        return  _input   # do nothing
    if split == 1:
        return  [item for sublist in _input for item in sublist]

def Input_training_data():
    encoder_inputs_conversation = u"training_data/encoder_inputs_conversation_sister_spec.json"
    decoder_targets_conversation = u"training_data/decoder_targets_conversation_sister_spec.json"
    print("preparing encoder input/ decoder targets...")
    print("Eecoder data : {} \nDecoder data : {}".format(encoder_inputs_conversation,
                                                         decoder_targets_conversation))
    with open(encoder_inputs_conversation, "r", encoding='utf-8') as jsonfile:
        encoder_inputs_data = json.load(jsonfile)["encoder_inputs_data"]
    with open(decoder_targets_conversation, "r", encoding='utf-8') as jsonfile:
        decoder_targets_data = json.load(jsonfile)["decoder_targets_data"]
    encoder_inputs_data = flatten(encoder_inputs_data, split=0)
    encoder_inputs_data = encoder_inputs_data[300:]
    decoder_targets_data = flatten(decoder_targets_data, split=0)
    decoder_targets_data = decoder_targets_data[300:]
    print("Training data loading completely!!")
    print("len() of the data : ", len(encoder_inputs_data))
    return encoder_inputs_data, decoder_targets_data, len(encoder_inputs_data)

def Input_dictionary_and_Embedding():
    print("preparing dictionary and embedding ...")
    GO_token = 40849
    End_token = 40850
    dictionary_in_json = u"model_repo/Vec200_40847/one_hot_dictionary.json"
    pretrained_embedding_matrix_in_json = u"model_repo/Vec200_40847/Embedding_matrix_only.json"
    with open(dictionary_in_json, "r", encoding='utf-8') as jsonfile:
        json_load = json.load(jsonfile)
        dictionary = json_load['dictionary']
        _reversed_dictionary = json_load["reversed_dictionary"]
    with open(pretrained_embedding_matrix_in_json, "r", encoding='utf-8') as jsonfile:
        Embedding = json.load(jsonfile)["embedding_matrix"]  #huge file...
        Embedding.append([0 for GO in range(200)])         #Add <GO> token
        Embedding.append([0 for End in range(200)])        #Add <EOS> token
    reverse_dictionary = {}
    for key, value in _reversed_dictionary.items():
        reverse_dictionary[int(key)] = value
    dictionary['<GO>'] = GO_token
    dictionary['<EOS>'] = End_token
    reverse_dictionary[GO_token] = '<GO>'
    reverse_dictionary[End_token] = '<EOS>'
    print("dictionary and embedding loading completely!!")
    return dictionary, reverse_dictionary, Embedding

def token_adder(idx, depth, start_token, end_token, mode="EOS"):
    """
    - EOS  mode :[[20],[43],[23],[0],[0]] --> [[20],[43],[23],[1],[0],[0]]
    - Both mode :[[20],[43],[23],[0],[0]] --> [[2],[20],[43],[23],[1],[0]]
    """
    add_token_idx = []
    idxcopy = idx.copy()
    for sentence in idxcopy:
        sentence_copy = sentence.copy()
        if mode == "Both":
            sentence_copy.insert(0, [start_token])
            sentence_copy = sentence_copy[:-1]
        add_token_sentence = []
        added = False
        # create a reversed list
        for num, char in enumerate(reversed(sentence_copy)):
            # some sentences can be overlength.
            if num == 0 and added == False:
                if char[0] != 0:
                    add_token_sentence.append([end_token])
                    added = True
                add_token_sentence.append(char)

            if num != 0:
                if char[0] == 0:
                    add_token_sentence.append(char)
                if char[0] != 0  and added == False:
                    add_token_sentence.append([end_token])
                    added = True
                if added == True:
                    add_token_sentence.append(char)
        # if it is an empty sentence
        if added == False:
            add_token_sentence.append([end_token])
        add_token_idx.append(list(reversed(add_token_sentence)))
    idxcopy = np.array(add_token_idx) # arrayize
    output_onehot = np.zeros((idxcopy.shape[1], idxcopy.shape[0], depth))
    for timestep   in range(idxcopy.shape[1]):
        for batch in range(idxcopy.shape[0]):
            max_idx = idxcopy[batch][timestep]
            output_onehot[timestep][batch][max_idx] = 1
    return output_onehot

def input_helper(encoder_inputs_data, decoder_targets_data,
                 start, end, depth, start_token, end_token):
    """
    add start token, end token.
    return input, decoder input, and decoder target
    """
    inputdata = token_adder(encoder_inputs_data[start:end],
                            depth, start_token, end_token)
    decoder_targetdata = token_adder(decoder_targets_data[start:end],
                                     depth, start_token, end_token)
    decoder_inputdata = token_adder(decoder_targets_data[start:end],
                                    depth, start_token, end_token, mode="Both")
    return inputdata, decoder_inputdata, decoder_targetdata

def ThreeD_onehot(idx, depth):
    output_onehot = np.zeros((idx.shape[0], idx.shape[1], depth))
    for timestep in range(len(idx)):
        for this_batch in range(len(idx[0])):
            max_idx = idx[timestep][this_batch]
            output_onehot[timestep][this_batch][max_idx] = 1
    return output_onehot
