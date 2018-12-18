# Attention mechanism
Based on paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Value-key attention model is used in this model. [see algorithm](https://github.com/hchungdelta/Simple_NN_API/tree/master/NN_v3.02_SandGlass/introduction/attention_mechanism/algorithm)
/ [see code](https://github.com/hchungdelta/Simple_NN_API/blob/master/NN_v3.02_SandGlass/ML/Layer/Attention.py)


The original model is as following.

<img src="https://github.com/hchungdelta/Simple_NN_API/blob/master/NN_v3.02_SandGlass/introduction/attention_mechanism/sandglass_base.gif" width="500">

Attention can be added on each layer, in this case, 5 attention layers are added.

<img src="https://github.com/hchungdelta/Simple_NN_API/blob/master/NN_v3.02_SandGlass/introduction/attention_mechanism/sandglass_attn.gif" width="500">

For a sanity check, I test whether the model can reverse the input as the target. For example,

```
input = [1, 4, 3, 2, 0, 0]
target = [0, 0, 2, 3, 4, 1]
```

The result is intriguing. The following gif file shows the alpha of each attention layer.

<img src="https://github.com/hchungdelta/Simple_NN_API/blob/master/NN_v3.02_SandGlass/introduction/attention_mechanism/attention_mechanism.gif" width="500">

 
| Attention layer   |   scope  |            description   | 
| :---:             |   :---:  |            :---:         | 
| 1        |   1 word     |   analyzing input in character-based manner | 
| 2 a      |   4 words    |   analyzing input in 4-character-based manner  |
| 2 b      |   1 word     |  analyzing input in character-based manner (further amplify the difference) | 
| 3        |    6 words   |  analyzing input on higher dimensions     | 
| 4        |    9 words   |  analyzing input on highest dimensions     | 

This model can implement attention mechanisms on different scopes (character-based, sub-sentence based ... etc), which is promising for analyzing the corpus. ( analyzing sentence structure on high-dimension attention layer, word-to-word translation on low-dimension attention layer.) 
