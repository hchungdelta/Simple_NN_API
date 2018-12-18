# Attention mechanism
Based on the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Value-key attention model is implemented in this model. [see algorithm](https://github.com/hchungdelta/Simple_NN_API/tree/master/NN_v3.02_SandGlass/introduction/attention_mechanism/algorithm)
/ [see code](https://github.com/hchungdelta/Simple_NN_API/blob/master/NN_v3.02_SandGlass/ML/Layer/Attention.py)


The original model is as following.

<img src="https://github.com/hchungdelta/Simple_NN_API/blob/master/NN_v3.02_SandGlass/introduction/attention_mechanism/sandglass_base.gif" width="500">

Attention can be added on each layer, in this case, 5 attention layers are added.

<img src="https://github.com/hchungdelta/Simple_NN_API/blob/master/NN_v3.02_SandGlass/introduction/attention_mechanism/sandglass_attn.gif" width="700">

For a sanity check, I test whether the model can reverse the input as the target. For example,

```
input = [1, 4, 3, 2, 0, 0]
target = [0, 0, 2, 3, 4, 1]
```

The result is intriguing. The following gif file shows the alpha of each attention layer.

<img src="https://github.com/hchungdelta/Simple_NN_API/blob/master/NN_v3.02_SandGlass/introduction/attention_mechanism/attention_mechanism.gif" width="500">

 
| Attention layer   |   scope  |            description   | 
| :---:             |   :---:  |            :---:         | 
| 1        |   1 word     |   character-based attention    | 
| 2 a      |   4 words    |   local corrlation attention  |
| 2 b      |   1 word     |  character-based attention   | 
| 3        |    6 words   |  local corrlation attention  | 
| 4        |    9 words   |  local corrlation attention  | 

* **Attention layer 1**: analyzing the input in a character-based manner, one can see that the same input has the same attention tendency (such as input number 5 and 23 in the above diagram)
* **Attention layer 2a**: analyzing the input considering local correlation (4 words-based), this layer works really well in this case, the diagram shows a clear one-to-one correspondence.
* **Attention layer 2b**: Similar to attention layer 1, while it amplifies the difference between each character.
* **Attention layer 3**: analyzing the input with a larger scope (6 words-based), however, in this case it is not necessary.
* **Attention layer 4**: analyzing the input with the largest scope, the "sentence structure" is obviously shown in this layer.

This model can implement attention mechanisms on different scopes (character-based, sub-sentence based ... etc), which is promising for analyzing the corpus. ( analyzing sentence structure on high-dimension attention layer, word-to-word translation on low-dimension attention layer.) 
