import numpy as np
import time

#a=np.random.randint(2,4 ,size=( 2, 5))

def Random_input_generator(batchsize, vocab_lower_limit, vocab_upper_limit, length_lower_limit, length_upper_limit):
    random_list = []
    for this_batch in range(batchsize):
        random_length = np.random.randint(length_lower_limit, length_upper_limit)
        this_array = np.random.randint(vocab_lower_limit, vocab_upper_limit, size=(random_length))
        random_list.append(this_array)
    return random_list

def ThreeD_onehot(array, vocabsize):
    output = np.zeros((array.shape[0], array.shape[1], vocabsize))
    for timestep in range(array.shape[0]):
        for batch in range(array.shape[1]):
            output[timestep, batch, array[timestep,batch]] = 1
    return output
def input_helper(random_list, length_upper_limit):
    """
    return encoder input, decoder target, and decoder input  with shape :(timstep, batch)
    <PAD> = 0
    <EOS> = 1
    <GO> = 2
    """
    encoder_input = []
    decoder_target = []
    decoder_input = []

    for batch in range(len(random_list)):
        encoder_input_candidate = random_list[batch]
        decoder_target_candidate = random_list[batch]
        decoder_input_candidate = random_list[batch]

        # add <EOS> and <GO>
        encoder_input_candidate = np.append(encoder_input_candidate, 1)
        decoder_target_candidate = np.append(decoder_target_candidate, 1)
        decoder_input_candidate = np.append(decoder_input_candidate, 1)
        decoder_input_candidate = np.insert(decoder_input_candidate, 0, 2)

        # add <PAD>
        while len(encoder_input_candidate) < length_upper_limit:
            encoder_input_candidate = np.append(encoder_input_candidate  , 0)
            decoder_target_candidate = np.append(decoder_target_candidate , 0)
            decoder_input_candidate = np.append(decoder_input_candidate  , 0)

        # add to list
        encoder_input.append(encoder_input_candidate)
        decoder_target.append(decoder_target_candidate)
        decoder_input.append(decoder_input_candidate)

    encoder_input = np.array(encoder_input).transpose(1, 0)
    decoder_target = np.array(decoder_target).transpose(1, 0)
    decoder_input = np.array(decoder_input).transpose(1, 0)
    return  encoder_input, decoder_target, decoder_input


