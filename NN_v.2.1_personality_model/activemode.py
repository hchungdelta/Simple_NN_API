"""
Active mode for testing trained model.
Author: HaoChien Hung.
Date: 01/09/2018 (MM/DD/YYYY)
"""
import numpy as np
import jieba
import ML
from data_importer import decode, encode, spec_vocab_searcher, ThreeD_onehot
from data_importer import Input_dictionary_and_Embedding
# mpi4py
comm = ML.EndeModel.comm
rank = ML.EndeModel.rank
size = ML.EndeModel.size

# set print options (optional)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# input data
dictionary, reversed_dictionary, _embeddings = Input_dictionary_and_Embedding()
amount_vocab = len(dictionary)
vocab_vector_size = len(_embeddings[0])

# W2V
_embeddings = [float(i) for vector in _embeddings  for i in vector]
embeddings = np.reshape(_embeddings, [amount_vocab, vocab_vector_size]).astype(np.float32)

# save path
savepath = 'data/0304copy.pickle'

# start / end token
GO_TOKEN = 40849
END_TOKEN = 40850


train_mode = 'infer'
display_epoch = 1
batch = 1 

length = 21  # max time step
hidden_units = 800
attn_head_amount = 1
attn_depth = 200
ortho_init = False
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
if EndeModel.comm != None:
    EndeModel.Bcast_Wb(initial=True)

EndeModel.Restore(savepath)
test_inputs = ["ただいま。",
               "僕と一緒に行きましょう？",
               "大好きです！",
               "学校はどうだった？",
               "妹が大好きです！",
               "大好きです。",
               "おはよう！",
               "ちょっと待って！",
               "お兄ちゃんのこと、どう思う？",
               "私と付き合ってください！",
               "どこで会います？",
               "明日どこへ行く？",
               "はやく、遅刻するぞ！"]
for character in ["general", "sister"]:
    print("character mode:", character)
    for a in range(len(test_inputs)):
        keyword_keyin = spec_vocab_searcher(test_inputs[a], dictionary)
        # encode the sentence
        encoded_keyin = encode(keyword_keyin, dictionary)
        encoded_keyin.extend([END_TOKEN])
        if len(encoded_keyin) < length:
            encoded_keyin.extend([0]*(length-len(encoded_keyin)))
        print("sample",a)
        #print(encoded_keyin)
        # reshape the sentence
        encoder_input = np.array(encoded_keyin).reshape(1, -1)
        # reshape the sentence
        encoder_input = ThreeD_onehot(encoder_input, amount_vocab).transpose(1, 0, 2)
        # for infer; persona_embedding in shape (batch, depth)
        if character == "general":
            persona_embedding = np.zeros((batch, persona_embed_depth)).astype(np.float32)
        if character == "sister":
            persona_embedding = np.ones((batch, persona_embed_depth)).astype(np.float32)


        encoder_input = encoder_input.astype(np.float32)
        decoder_init = np.zeros((batch, amount_vocab)).astype(np.float32)
        decoder_init[:, GO_TOKEN] = 1
        pred, _ = EndeModel.Forward(encoder_input, decoder_init, None,[length, length], persona_embedding)
        EndeModel.Timestep_gather()
    
        # decode part
        INPUT = np.argmax(encoder_input, axis=2).transpose(1, 0)
        PRED = np.argmax(pred, axis=2).transpose(1, 0)

        print("INPUT  : ", decode(INPUT[0], reversed_dictionary))
        print("PRED   : ", decode(PRED[0], reversed_dictionary))
