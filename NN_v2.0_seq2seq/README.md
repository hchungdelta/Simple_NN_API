# Seq2Seq model in numpy

LSTM sequence encoder-decoder model in numpy.

## environment :
* Python 3.5
* Numpy 1.14
* numba 0.39.0
* (optional) mpi4py

## .py file:
* data_importer.py: (subsidiary code, for inputing dictionary, data, and some functions.) 
* seq2seq.py: Training mode. (due to privacy issue, training data isn't uploaded.)
* tester.py: interactive test mode. (Infer mode)
#### nomenclature are descripted in sub-directory "ML"

## training data:
training data source: [小説家になろう](https://syosetu.com/)
#### conversation training data: 
1. 80,000 sentences (specific character conversation: younger sister(妹属性))
2. 660,000 sentences (specific character conversation: general)

#### word2vec training data (Using genism):
Word2Vec training data: 200 MB data  

## computation cost:
1 CPU, 1 week training.

# Project:

## data preparing
1. dictionary: words simple statistical method. 
[introduction/define_vocabulary](https://github.com/hchungdelta/Simple_NN_API/tree/master/NN_v2.0_seq2seq/introduction/define_vocabulary) 


2. training data (conversation).
Extracting the conversation from novels (source: [小説家になろう](https://syosetu.com/)).
In many circumstances, two sentences may not be related. For example:


Solution : set up a threshold for "Word distance" between sentences in the novel.






