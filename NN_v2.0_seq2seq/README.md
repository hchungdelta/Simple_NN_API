# Seq2Seq model in numpy

LSTM sequence encoder-decoder model in numpy.

## environment :
* Python 3.5
* Numpy 1.14
* numba 0.39.0
* (optional) mpi4py

## .py file:
* data_importer.py: subsidiary code, for inputing dictionary, data, and some functions.
* seq2seq.py: training mode. (due to privacy issue, training data isn't uploaded.)
* tester.py: interactive test mode. (Infer mode)
#### nomenclature are descripted in sub-directory "ML"

## training data:
training data source: [小説家になろう](https://syosetu.com/)
*  conversation training data: 
1. 80,000 sentences (specific character conversation: younger sister(妹属性))
2. 660,000 sentences (specific character conversation: general)

* word2vec training data (Using genism):
1. 200 MB txt data  

## computation cost:
* 1 CPU, 1 week training.

# Project:

## data preparing
1. dictionary: words simple statistical method. 

[see introduction/define_vocabulary](https://github.com/hchungdelta/Simple_NN_API/tree/master/NN_v2.0_seq2seq/introduction/define_vocabulary) 


2. training data : conversations from novel.
Extracting the conversations from novels  

[see introduction/extract_conversation](https://github.com/hchungdelta/Simple_NN_API/tree/master/NN_v2.0_seq2seq/introduction/extract_conversation)

3. character-specific conversation: 
Extracting character-specific conversations from all conversations  

[see introduction/extract_character_specific_conversation](https://github.com/hchungdelta/Simple_NN_API/tree/master/NN_v2.0_seq2seq/introduction/extract_character_specific_conversation)



## LSTM mechanism


## training model

An embarrassingly parallel are implemented automatically in this API (if available).
In short, 
each processor uses the same weight and bias at each step, while using different data to train them. (e.g. processor1 uses data batch 1-32, processor2 uses bata batch 33-64 ...etc.) And latter sum up the dW and db from each processor for updating.





