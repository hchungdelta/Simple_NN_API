# Define vocabulary

Unlike English, commonly in the Japanese language(and also Chinese), the sentence is constructed by word in consecutive order, without space to separate the word.
For example:


```
English: I went to the office, but no one was there
日本語: 私はオフィスへ行ったけど、誰もいなかった
```
If we write this sentence In English vein, it would look like:
```
日本語:-私-は-オフィス-へ-行った-けど、誰-も-いなかった
```

In English, the most straightforward approach to construct an dictionary is to one-hot the vocabulary, and then construct a word embedding based on it for later usage.  (of course, there are some problems inherited in English, many new mechanisms are suggested to overcome these issue such as [Character-based method](https://arxiv.org/abs/1511.04586).)

It indicates that the vocabulary cannot be easily distinguished from a sentence without a basic understanding of the Japanese language.
Hence we need to contstruct/define a dictionary first.
Many vocabulary-separated mechanisms have been developed such as [mecab](http://taku910.github.io/mecab/), which can reach high accuracy in parsing sentence. However, here I want to try an idea, *is human-like grammar analysis to be necessary?*

## Statistical method:

This is a naive idea for vocabulary-searching. Aspiring by the fundamental idea of word2vec, I write a code to record the frequency of each candidate vocabulary, if a combination of individual characters to be of high frequency, it is likely to be a defined vocabulary. In practice : 

<img src="dict_record.gif" width="450">

<img src="dict_record2.gif" width="450">


200MB txt novel data are used for vocabulary-searching, I set the upper limit of the vocabulary length to be 7. and if the frequency of certain combination is larger than 120, it will be registered in dictionary.

There are three issues arise in this method. 
* duplicate information
* lengthy words
* postpositional particle　格助詞「て、に、を、は、が」

#### duplicate information
It is necessary to delete "duplicate information", for example, it is easy to imagine that in this method the dictionary will register the following vocabulary.
```
オフィス  (like office)  --> true vocabulary
フィス    (like   fice)  --> false vocabulary
ィス      (like    ice)  --> false vocabulary
```

However, This can be easily solved by check the frequency, since characters are coupled together, so they share a similar frequency. 
The idea is as follows.
```
if word2 in word1:   # word2 is a sub-vocabulary of word1
    if tolerant_rate1*frequency(word2) > frequency(word1) > tolerant_rate2*frequency(word2):
        expr(word2_replace_word1)
# tolerant_rate1 = 1.1
# tolerant_rate2 = 0.9
```
In this case, オフィス will replace the フィス and ィス.

#### lengthy words
