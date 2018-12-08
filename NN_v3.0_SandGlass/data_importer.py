"""
For preparing training data.

Author: Hao Chien, Hung.
Date : 5/12/2018
"""
import json
import numpy as np

def encode(input_str, dicts):
    sentence = input_str.split(" ")
    encoded_sentence = []
    for word in sentence:
        if  word != "" and len(sentence) > 1:
            try:
                encoded_sentence.append(dicts[word])
            except:
                encoded_sentence.append(dicts["<UNK>"])
    return encoded_sentence

def decode(input_list, reversed_dicts):
    decoded_sentence = ""
    for idx in input_list:
        try:
            add_word = reversed_dicts[idx]
            if add_word != "<PAD>":
                decoded_sentence += reversed_dicts[idx]+ " "
        except:
            decoded_sentence += "--BEEP--"

    return decoded_sentence

def encode_data(data, dicts):
    conversations = data.split('\n')[:-1]
    encoded_conv = []
    for conversation in conversations:
        encoded_conv.append(encode(conversation, dicts))
    inputs = encoded_conv[:-1]
    targets = encoded_conv[1:]
    trainingdata = {"inputs":inputs, "targets":targets}
    with open("traininghortata.json", 'w') as jsonfile:
        json.dump(trainingdata, jsonfile)

def swapper(array, swap_amount):
    len_array = len(array)
    for this_swap in range(swap_amount):
        swap_a_idx = np.random.randint(len_array)
        swap_b_idx = np.random.randint(len_array)
        swap_a = array[swap_a_idx].copy()
        swap_b = array[swap_b_idx].copy()
        array[swap_a_idx], array[swap_b_idx] = swap_b, swap_a
    return array




offset = 0
def batch_helper(inputdata, targetdata, batch, lengthlimit, mode="cut"):
    global offset
    batchinput = []
    batchtarget = []
    batch_decode_input = []
    batchinput_length = []
    batchtarget_length = []

    for this_idx in range(batch):
        if offset == len(inputdata):
            offset = 0
        if len(inputdata[offset]) >= lengthlimit-2:
            batchinput.append(inputdata[offset][:lengthlimit-2])
        else:
            batchinput.append(inputdata[offset].copy())

        if len(targetdata[offset]) >= lengthlimit-2:
            batchtarget.append(targetdata[offset][:lengthlimit-2].copy())
            batch_decode_input.append(targetdata[offset][:lengthlimit-2].copy())
        else:
            batchtarget.append(targetdata[offset].copy())
            batch_decode_input.append(targetdata[offset].copy())
        offset += 1
    for this_idx in range(batch):
        batchinput[this_idx].append(1)
        batchtarget[this_idx].append(1)
        batch_decode_input[this_idx].append(1)
        batch_decode_input[this_idx].insert(0, 2)
        batchinput_length.append(len(batchinput[this_idx]))
        batchtarget_length.append(len(batchtarget[this_idx]))
    batchinput_maxlength = np.max(batchinput_length)
    batchtarget_maxlength = np.max(batchtarget_length)
    for this_idx in range(batch):
        if mode == "nocut":
            batchinput_cutoff = lengthlimit
            batchtarget_cutoff = lengthlimit
        if mode == 'cut':
            batchinput_cutoff = batchinput_maxlength
            batchtarget_cutoff = batchtarget_maxlength+1
        if batchinput_length[this_idx] < batchinput_cutoff:
            need_to_add_pad_amount = batchinput_cutoff - batchinput_length[this_idx]
            batchinput[this_idx].extend([0] * need_to_add_pad_amount)

        if batchtarget_length[this_idx] < batchtarget_cutoff:
            need_to_add_pad_amount = batchtarget_cutoff  -  batchtarget_length[this_idx]
            batchtarget[this_idx].extend([0] * need_to_add_pad_amount)
            batch_decode_input[this_idx].extend([0] * (need_to_add_pad_amount-1))

    return batchinput, batch_decode_input, batchtarget

def input_helper(input_data, vocab_size):
    """
    helper for input_randomer
    output as onehot with shape ( timestep, batch, depth)
    """
    arrayize = np.array(input_data)
    change_shape = np.transpose(arrayize, (1, 0))
    output = ThreeD_onehot(change_shape, vocab_size)
    return output

def ThreeD_onehot(idx, depth):
    """
    for input with shape (timestep x batch x depth )
    """
    output_onehot = np.zeros((idx.shape[0], idx.shape[1], depth))
    for timestep in range(len(idx)):
        for batch in range(len(idx[0])):
            max_idx = idx[timestep][batch]
            output_onehot[timestep][batch][max_idx] = 1

    return output_onehot

def input_dict(dict_file_name):
    with open(dict_file_name, 'r', encoding='utf-8') as jsonfile:
        read_dicts = json.load(jsonfile)
        dicts = read_dicts["dictionary"]
        _reversed_dicts = read_dicts["reversed_dictionary"]
        reversed_dicts = dict()
        for key, value in _reversed_dicts.items():
            reversed_dicts[int(key)] = value
    return dicts, reversed_dicts

def input_training_data(data_file_name):
    with open(data_file_name, 'r') as jsonfile:
        training_data = json.load(jsonfile)
        inputdata = training_data["inputs"]
        targetdata = training_data["targets"]
    return inputdata, targetdata
