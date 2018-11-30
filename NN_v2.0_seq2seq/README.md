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
* seq2seq training data : 
1. 80,000 sentences from website : 小説家になろう (specific character conversation: younger sister(妹属性))
2. 660,000 sentences from　website : 小説家になろう (specific character conversation: general)
(2. is still training)
* Word2Vec (Using genism) training data: 200 MB data from website :小説家になろう

## computation cost:
1 CPU, 1 week training.

## data preparing:
1. dictionary:word 
statisctial method

2. sentence:
