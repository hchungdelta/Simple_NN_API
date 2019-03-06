import numpy as np
import time
import ML 
from data_importer import input_helper, decode
from data_importer import Input_training_data, Input_dictionary_and_Embedding

# mpi4py
comm = ML.EndeModel.comm
rank = ML.EndeModel.rank
size = ML.EndeModel.size

# set print options (optional)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# record the cost of time
starttime=time.time()

# save path for trainable variables 
savepath = 'data/0108.pickle'

# start / end token
GO_TOKEN = 40849
END_TOKEN = 40850

# input data
dictionary, reverse_dictionary, _embeddings = Input_dictionary_and_Embedding()
encoder_inputs_data, decoder_targets_data, persona_label = Input_training_data()
amount_vocab = len(dictionary)
vocab_vector_size = len(_embeddings[0])

# W2V
_embeddings = [float(i) for vector in _embeddings  for i in vector]
embeddings = np.reshape(_embeddings, [amount_vocab, vocab_vector_size]).astype(np.float32)

# parameter
train_mode = 'train'
display_epoch = 25
batch = 64 


length = 21  # max time step
hidden_units = 800 
attn_head_amount = 1
attn_depth = 200
ortho_init = True
use_LN = 2 
persona_embed_depth = 100


# timestep, batch, input_units, output_units
encoder_info1 = [length ,batch, vocab_vector_size, hidden_units]   
decoder_info1 = [length ,batch, vocab_vector_size, 2*hidden_units]

# word2vec layer
Embedding_matrix = ML.Layer.FCL_layer.Embedding(embeddings)

#(encoder) Bi LSTM
fw_Encoder1 = ML.Layer.LSTM_layer.LSTMcell(encoder_info1, output_form ="All", ortho=ortho_init, LSTM_LN=use_LN)
bw_Encoder1 = ML.Layer.LSTM_layer.LSTMcell(encoder_info1, output_form ="All", ortho=ortho_init, LSTM_LN=use_LN)
Bi_Encoder1 = ML.Layer.LSTM_layer.BiLSTM(fw_Encoder1,bw_Encoder1, LSTM_LN=use_LN)

#(decoder) LSTM
Decoder1 = ML.Layer.LSTM_layer.LSTMcell(decoder_info1,output_form="All",ortho=ortho_init, LSTM_LN=use_LN)

# Fully connected layer/tanh
en_FCLlayer = ML.Layer.FCL_layer.annexed_timestep_xW_b((2*hidden_units, attn_depth),ortho=ortho_init)
partial_encoder_tanh = ML.Layer.Acti_layer.partial_Tanh(2*hidden_units)

de_FCLlayer = ML.Layer.FCL_layer.timestep_xW_b((4*hidden_units-attn_depth+persona_embed_depth, amount_vocab), ortho=ortho_init)

# softmax corss entropy
softmax_cross_entropy = ML.Layer.FCL_layer.softmax_cross_entropy(train_mode)

# attention
single_value_depth = int(2*hidden_units/attn_head_amount)
single_key_depth = int(attn_depth/attn_head_amount)
MultiAttn = ML.Layer.Attention.LSTM_MultiAttn(attn_head_amount, single_value_depth, single_key_depth)

# concatenate
persona_concat = ML.Layer.LSTM_layer.persona_concat()

# architecture
EndeModel = ML.EndeModel.Model(lr=0.0016, optimizer='adam', mode=train_mode, clipping=True, clip_value=1.30)
EndeModel.add(Embedding_matrix, belong_to="Encoder")
EndeModel.add(Bi_Encoder1, belong_to="Encoder", connection_label=1)
EndeModel.add(en_FCLlayer, belong_to="Encoder")
EndeModel.add(partial_encoder_tanh, belong_to="Encoder")
EndeModel.add(MultiAttn, belong_to="Encoder")

EndeModel.add(Embedding_matrix, belong_to="Decoder")
EndeModel.add(Decoder1, belong_to="Decoder",connection_label=1)
EndeModel.add(MultiAttn, belong_to="Decoder")
EndeModel.add(persona_concat, belong_to="Decoder")
EndeModel.add(de_FCLlayer, belong_to="Decoder")
EndeModel.add(softmax_cross_entropy, belong_to="Decoder")
# initializer
if EndeModel.comm != None :
    EndeModel.Bcast_Wb(initial=True)

EndeModel.Restore('data/0108.pickle')

displaytime = time.time()
total_L = 0
offset = 0
for a in range(int(60000)):
    print("step",a)
    if offset+(rank+1)*batch >len(encoder_inputs_data):
        offset = 0
    batchinput, batch_decode_input, batchtarget = input_helper(encoder_inputs_data,
                                                               decoder_targets_data,
                                                               offset+rank*batch,
                                                               offset+(rank+1)*batch,
                                                               amount_vocab,
                                                               GO_TOKEN,
                                                               END_TOKEN)
    this_persona_label = persona_label[offset+rank*batch:offset+(rank+1)*batch]
    persona_embedding = np.zeros((length, batch, persona_embed_depth)).astype(np.float32)
    for idx, this_batch_label in enumerate(this_persona_label):
        if this_batch_label == 1:
            persona_embedding[:, idx, :] = 1
    
    offset += batch*size
    en_cutoff_length = length
    de_cutoff_length = length
    batchinput = batchinput.astype(np.float32)
    batch_decode_input = batch_decode_input.astype(np.float32)
    target = batchtarget.astype(np.float32)

    pred, Loss = EndeModel.Forward(batchinput, batch_decode_input,
                                   target,[en_cutoff_length, de_cutoff_length],
                                   persona_embedding)
    EndeModel.Backprop()

    total_L += Loss/display_epoch
    if a% display_epoch  ==0 and rank == 0:
        print("Loss {:.5f}".format(total_L))
        alpha = MultiAttn.get_alpha()
        PRED = np.argmax(pred, axis=2)
        INPUT = np.argmax(batchinput, axis=2)
        TARGET = np.argmax(batchtarget, axis=2)
        #DEINPUT = np.argmax(batch_decode_input, axis=2)
        INPUT = INPUT.transpose(1, 0)
        #DEINPUT = DEINPUT.transpose(1, 0)
        PRED = PRED.transpose(1, 0)
        TARGET = TARGET.transpose(1, 0)
        for number in range(3):
            print(number)
            print("alpha",alpha)
            print("INPUT  : ", decode(INPUT[number], reverse_dictionary))
            print("PRED   : ", decode(PRED[number], reverse_dictionary))
            print("TARGET : ", decode(TARGET[number], reverse_dictionary))


        total_L = 0
        print("time cost {:.2f} seconds ,for one display epoch".format(
            time.time()-displaytime))
        displaytime = time.time()
        EndeModel.Save(savepath)




