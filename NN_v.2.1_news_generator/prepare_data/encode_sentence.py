import numpy as np
import collections
import json
#all_description.txt  all_headline.txt
with open("corpus/description.txt", 'r') as txtfile :
    data = txtfile.readlines()
with open("corpus_json/dictionary.json" , 'r', encoding='utf-8') as jsonfile :
    read_dicts = json.load(jsonfile)
    dicts =   read_dicts["dictionary"]
    _reversed_dicts =  read_dicts["reversed_dictionary"]
    reversed_dicts = dict()
    for key, value in _reversed_dicts.items(): 
        reversed_dicts[int(key)] = value   # json 

#with open("trainingdata.json" , 'r')  as jsonfile:
#    training_data=json.load(jsonfile)
#    inputdata = training_data["inputs"]
#    targetdata= training_data["targets"]

def encode(input_str):
    sentence = input_str.split(" ")
    encoded_sentence = []
    for word in sentence :
        if  word != "" and len(sentence) > 1:
            try:
                encoded_sentence.append(dicts[word] )  
            except :
                encoded_sentence.append(dicts["unknown"])
    return encoded_sentence

def decode(input_list):
    decoded_sentence = ""
    for idx in input_list :
        if idx ==3:
            decoded_sentence += "<UNK>"
        else:
            decoded_sentence += reversed_dicts[idx]+ " " 
        
    return decoded_sentence
 

def encode_data(data):
     
    encoded_conv = []
    for conversation in data:
        conversation = conversation.split(':')[1]
        encoded_conv.append(encode(conversation)[:-1])
    inputs = encoded_conv
    trainingdata={"inputs":inputs}
    with open("corpus_json/description.json" , 'w')  as jsonfile:
        training_data=json.dump(trainingdata,jsonfile)
encode_data(data)
#with open("corpus_json/description.json" , 'r')  as jsonfile:
#    desc=json.load(jsonfile)['inputs']
#with open("corpus_json/headline.json" , 'r')  as jsonfile:
#    head=json.load(jsonfile)['inputs']

