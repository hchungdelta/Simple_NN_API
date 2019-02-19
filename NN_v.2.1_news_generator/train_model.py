import time
import numpy as np
import ML
from data_importer import decode, input_Embedding
from data_importer import input_training_data, input_dict, input_helper, batch_helper


# mpi4py
comm = ML.EndeModel.comm
rank = ML.EndeModel.rank
size = ML.EndeModel.size

# set print options (optional)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)



encode_input_file = "corpus_json/headline.json"
decode_target_file = "corpus_json/description.json"
dictionary_file = "corpus_json/dictionary.json"
inputdata, targetdata = input_training_data(encode_input_file, decode_target_file)
dicts, reversed_dicts = input_dict(dictionary_file)

embedding_path = "corpus_json/embedding.json"
embeddings = input_Embedding(embedding_path, dtype=np.float32)
#print(decode(decode_target[a],reversed_dicts))


# training data information
train_mode = 'train'
use_LN = 2 # 0:None, 1:simple mode (1 LN), 2:full mode (8 LNs)
ortho_init = True # to use orthogonal matrix initializer or not.
batch = 32
output_length = 90
input_length = 24
vocab_vector_size = 300
amount_vocab = len(dicts)
amount_of_printout = 3
display = 20
# save path
savepath = 'data/1208.pickle'

# parameter
hidden_units = 900
Attn_depth = 300

DotAttn = ML.Layer.Attention.DotAttn_3d()
Attn = ML.Layer.Attention.LSTM_Attn_helper(DotAttn, Attn_depth)
# timestep, batch, input_units, output_units
encoder_info1 = [input_length, batch, vocab_vector_size, hidden_units]
decoder_info1 = [output_length, batch, vocab_vector_size, 2*hidden_units]

# word2vec layer
Embedding_matrix = ML.Layer.FCL_layer.Embedding(embeddings)

#(encoder) Bi LSTM
fw_Encoder1 = ML.Layer.LSTM_layer.LSTMcell(encoder_info1, output_form="All",
                                           ortho=ortho_init, LSTM_LN=use_LN)
bw_Encoder1 = ML.Layer.LSTM_layer.LSTMcell(encoder_info1, output_form="All",
                                           ortho=ortho_init, LSTM_LN=use_LN)
Bi_Encoder1 = ML.Layer.LSTM_layer.BiLSTM(fw_Encoder1, bw_Encoder1, LSTM_LN=use_LN)

#(decoder) LSTM
Decoder1 = ML.Layer.LSTM_layer.LSTMcell(decoder_info1, output_form="All",
                                        ortho=ortho_init, LSTM_LN=use_LN)

# bcast the initial weights and bias from rank 0 to all the others
en_FCLlayer = ML.Layer.FCL_layer.annexed_timestep_xW_b((2*hidden_units, Attn_depth),
                                                       ortho=ortho_init)
partial_encoder_tanh = ML.Layer.Acti_layer.partial_Tanh(2*hidden_units)

de_FCLlayer = ML.Layer.FCL_layer.timestep_xW_b((4*hidden_units - Attn_depth, amount_vocab),
                                               ortho=ortho_init)
softmax_cross_entropy = ML.Layer.FCL_layer.softmax_cross_entropy(train_mode)

# architecture
EndeModel = ML.EndeModel.Model(lr=0.0016, optimizer='adam', mode=train_mode,
                               clipping=True, clip_value=1.40)
EndeModel.add(Embedding_matrix, belong_to="Encoder")
EndeModel.add(Bi_Encoder1, belong_to="Encoder", connection_label=1)
EndeModel.add(en_FCLlayer, belong_to="Encoder")
EndeModel.add(partial_encoder_tanh, belong_to="Encoder")
EndeModel.add(Attn, belong_to="Encoder")

EndeModel.add(Embedding_matrix, belong_to="Decoder")
EndeModel.add(Decoder1, belong_to="Decoder", connection_label=1)
EndeModel.add(Attn, belong_to="Decoder")
EndeModel.add(de_FCLlayer, belong_to="Decoder")
EndeModel.add(softmax_cross_entropy, belong_to="Decoder")


# initializer
if EndeModel.comm != None:
    EndeModel.Bcast_Wb(initial=True)

EndeModel.Restore(savepath)

tot_L = 0
# record the cost of time
display_time = time.time()

for step in range(22000):
    if rank == 0:
        batchinput, batch_decode_input, batchtarget = batch_helper(inputdata,
                                                                   targetdata,
                                                                   batch*size,
                                                                   output_length,
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
    batchinput = input_helper(batchinput, amount_vocab)
    batchtarget = input_helper(batchtarget, amount_vocab)
    batch_decode_input = input_helper(batch_decode_input, amount_vocab)
    feed_input = batchinput[:input_length]

    feed_input = feed_input.astype(np.float32)
    batch_decode_input = batch_decode_input.astype(np.float32)
    batchtarget = batchtarget.astype(np.float32)


    pred, Loss = EndeModel.Forward(feed_input, batch_decode_input, batchtarget,
                                   [input_length, output_length])
    EndeModel.Backprop()

    tot_L += Loss/display
    if rank == 0 and step%display == 0:
        print("Loss: {:.3f} ".format(tot_L))
        tot_L = 0
        ENINP = np.argmax(feed_input[:, :amount_of_printout, :], axis=2).transpose(1, 0)
        TARGET = np.argmax(batchtarget[:, :amount_of_printout, :], axis=2).transpose(1, 0)
        PRED = np.argmax(pred[:, :amount_of_printout, :], axis=2).transpose(1, 0)
        for number in range(amount_of_printout):
            print("sample ", number)
            print("input  : ", decode(ENINP[number].tolist(), reversed_dicts))
            print("target : ", decode(TARGET[number].tolist(), reversed_dicts))
            print("pred   : ", decode(PRED[number].tolist(), reversed_dicts))

        print("cost : {:.3f} seconds".format(time.time()-display_time))
        EndeModel.Save(savepath)
        display_time = time.time()
