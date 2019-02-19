import numpy as np
import collections
import json
with open("corpus/news_both_test.txt", 'r') as txtfile :
    data = txtfile.read()
#with open("dict.json" , 'r', encoding='utf-8') as jsonfile :
#    dicts = json.load(jsonfile)["dictionary"]


def tokenize(data,threshold):
    """ 
    Input data(txt) with unique word be separated by space.
    """
    data=data.replace("\n", "") 
    dictionary = dict()
    split_by_space = data.split(' ')
    frequency_recorder = collections.Counter(split_by_space)
    dictionary.update({"<PAD>":0,"<EOS>":1,"<GO>":2,"<UNK>":3})
    indexer = 4
    for  a in frequency_recorder.keys() :
        if  frequency_recorder[a] >= threshold :
            dictionary.update({a:indexer})
            indexer += 1
    reversed_dictionary = {v:k for k,v in dictionary.items() }
    return frequency_recorder , dictionary,reversed_dictionary
 


    
def make_dict(threshold):
    frequency,dictionary , reversed_dictionary= tokenize(data,threshold)
    saveme= {"frequency":frequency,"dictionary":dictionary,"reversed_dictionary":reversed_dictionary}
    with open("NEWS_dict_test.json","w", encoding='utf-8') as jsonfile:
        json.dump(saveme,jsonfile,ensure_ascii=False)


make_dict(2)
