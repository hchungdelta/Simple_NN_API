"""
Testing the trained model (news headline-content generator).
Author: HaoChien Hung.
Date: 02/22/2018 (MM/DD/YYYY)
"""
import numpy as np
import jieba
import ML
from data_importer import encode, decode, ThreeD_onehot, input_Embedding, input_dict

# mpi4py
comm = ML.EndeModel.comm
rank = ML.EndeModel.rank
size = ML.EndeModel.size

# set print options (optional)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# input dictionary and embeddings
dictionary_file = "corpus_json/dictionary.json"
dicts, reversed_dicts = input_dict(dictionary_file)

embedding_path = "corpus_json/embedding.json"
embeddings = input_Embedding(embedding_path, dtype=np.float32)

# save path
savepath = 'data/0219.pickle'

# parameters
batch = 1
input_length = 24 #input length(headline)
output_length = 90 #output length(description)
vocab_vector_size = 300
amount_vocab = len(dicts)
use_LN = 2 # 0 = None/1:single/2:full LN in LSTM
use_mode = 'infer' # train of infer

# parameter
hidden_units = 800
Attn_depth = 300

# timestep, batch, input_units, output_units
encoder_info1 = [input_length, batch, vocab_vector_size, hidden_units]
decoder_info1 = [output_length, batch, vocab_vector_size, 2*hidden_units]

# word2vec layer
Embedding_matrix = ML.Layer.FCL_layer.Embedding(embeddings)

#(encoder) Bi LSTM
fw_Encoder1 = ML.Layer.LSTM_layer.LSTMcell(encoder_info1, output_form="All", LSTM_LN=use_LN)
bw_Encoder1 = ML.Layer.LSTM_layer.LSTMcell(encoder_info1, output_form="All", LSTM_LN=use_LN)
Bi_Encoder1 = ML.Layer.LSTM_layer.BiLSTM(fw_Encoder1, bw_Encoder1, LSTM_LN=use_LN)

#(decoder) LSTM
Decoder1 = ML.Layer.LSTM_layer.LSTMcell(decoder_info1, output_form="All", LSTM_LN=use_LN)


# fully connected layer
en_FCLlayer = ML.Layer.FCL_layer.annexed_timestep_xW_b((2*hidden_units, Attn_depth))
partial_encoder_tanh = ML.Layer.Acti_layer.partial_Tanh(2*hidden_units)

de_FCLlayer = ML.Layer.FCL_layer.timestep_xW_b((4*hidden_units - Attn_depth, amount_vocab))
softmax_cross_entropy = ML.Layer.FCL_layer.softmax_cross_entropy(use_mode)

# Attention layer
DotAttn = ML.Layer.Attention.DotAttn_3d()
Attn = ML.Layer.Attention.LSTM_Attn_helper(DotAttn, Attn_depth)

# architecture (Encoder & decoder)
EndeModel = ML.EndeModel.Model(lr=0.0015, optimizer='adam', mode=use_mode)
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

# though jieba can help parse the context, it isn't accurate enough,
# one had better to add space between vocabluaries.
# you can try belowing headline samples
"""
桃園 機場 擬 建立 水上 樂園 推廣 觀光,
快訊／台中 女 老師 失蹤三日　警公布特徵齊協尋,
失業一年、妻分居談離婚　狠夫Po文「想離開心寒地帶」,
車前 突 現 人影！　女子暗夜過馬路嚇壞駕駛,
天氣一週「3 變」！明 稍 涼週四又回暖　週末再轉濕涼,
韓國瑜猛轟中央　呂秀蓮：半年後就原形畢露,
女大生頭痛昏迷4天　醒來已當媽驚喊：孩子不是我的！,
洗手台就是飯桌！清潔工躲廁所淒涼用餐　惹網友不捨,
燈會竟出現「1 8 禁」畫面？童好奇 問 爸：這在幹嘛,
「自殺大樓」一躍而下　女大生沾滿死者血…淒厲慘叫昏倒,
疑為趕搭公車　7 旬兄妹闖紅燈過馬路 遭 撞亡,
羅志祥無預警「全面退出」　粉絲驚呆爆 內幕,
7 年增 1 4 . 5 萬　桃園「人口淨遷入」冠居全台
"""
while True:
    # enter jieba parsing
    keyin = input("Input the headline:")
    if keyin == "exit()":
        break
    jieba_cut = jieba.cut(keyin)
    split_output = " ".join(jieba_cut)
    print("original input (jieba parse) :", split_output)

    # preparation before entering neural network.
    test_input = encode(split_output, dicts)
    test_input.append(1) # start token
    if len(test_input) < input_length:
        test_input.extend([0]*(input_length-len(test_input))) # zero paddings
    test_input = np.array([test_input])

    init_decoder = np.zeros((batch, amount_vocab)).astype(np.float32)
    init_decoder[:, 2] = 1
    # enter neural network architecture
    feed_input = ThreeD_onehot(test_input, amount_vocab).transpose(1, 0, 2)
    pred, L = EndeModel.Forward(feed_input, init_decoder, None, [input_length, output_length])
    EndeModel.Timestep_gather()

    ENINP = np.argmax(feed_input, axis=2).transpose(1, 0)
    PRED = np.argmax(pred, axis=2).transpose(1, 0)

    print("headline (input): ", decode(ENINP[0].tolist(), reversed_dicts))
    print("description (output): ", "".join(decode(PRED[0].tolist(), reversed_dicts)))
    print("=======================================================================")
