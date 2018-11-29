from gensim.models import word2vec
import json

model   = word2vec.Word2Vec.load('sis_tot_slight2.model')
W=model.wv.syn0

W2V_dict=dict()
word_vector_length =200
zero_space=[0] * word_vector_length
unknown=[0.5] * word_vector_length
W2V_dict[" "]= zero_space
W2V_dict["unknown"]= unknown
W2V_matrix_only=[]
W2V_matrix_only.append(zero_space)
W2V_matrix_only.append(unknown)


for word in model.wv.vocab: 
    single_W2V=  model[word].tolist()
    single_W2V = [ '%.6f' % elem for elem in single_W2V ]
    W2V_dict[word]=single_W2V
    W2V_matrix_only.append(single_W2V)

dictionary=dict()
reversed_dictionary=dict()
for index, key in enumerate(W2V_dict):
    dictionary[key]=index
    reversed_dictionary[index]=key

DICT= {"dictionary" : dictionary, "reversed_dictionary": reversed_dictionary }
W2V_matrix_only={"embedding_matrix": W2V_matrix_only}

with open("W2V_dict.json", 'w', encoding='utf-8'  ) as jsonfile :
    json.dump(W2V_dict,jsonfile, ensure_ascii=False)
with open("Embedding_matrix_only.json", 'w', encoding='utf-8'  ) as jsonfile :
    json.dump(W2V_matrix_only,jsonfile, ensure_ascii=False)
with open("one_hot_dictionary.json", 'w', encoding='utf-8'  ) as jsonfile :
    json.dump(DICT,jsonfile,  ensure_ascii=False)


