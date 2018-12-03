# SandGlass
<img src="introduction/sandglass.png" width="302">

* Autoencoder
* Inception
[Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke (2016)](https://ai.google/research/pubs/pub45169)
* Attention
[Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2014)](https://arxiv.org/abs/1409.0473)
* Batch Normalization
[Sergey Ioffe, Christian Szegedy (2015)](https://arxiv.org/abs/1502.03167)
* Batch Renormalization
[Sergey Ioffe (2017)](https://arxiv.org/abs/1702.03275)
## Basic idea:



| our brain       |      in neural network      |            description   |
| :---:           |           :---:             |            :---:         | 
|stimulus         |    encoder conv. part       | receive the information, translate into our brain. | 
| brain processing|center fully connected layer |    process the information. |
| response        |     decoder conv. part      |  verbalize the information. | 

## Pros:
- Compare to traditional seq2seq model, this mechanism is less likely to lose information.
- Easy to perform parallel computing, expected to be trained faster.
- Can be trained without decoder input. 
- Relatively robust, since the prediction is dependent on the whole corpus rather than highly depends on the previous output.

## Cons:
1. Complicated.


## Future work:



#### brain processing part
* Memory 
* personality (persona embedding)

#### stimulus & response part.
* [Transformer](https://arxiv.org/abs/1706.03762)
* [google bert](https://arxiv.org/pdf/1810.04805.pdf)


