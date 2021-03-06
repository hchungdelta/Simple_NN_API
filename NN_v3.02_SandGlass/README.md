# SandGlass
Since the neural network architecture of this model looks like sandglass, I temporarily call this model SandGlass.
[see overview for further information](https://github.com/hchungdelta/Simple_NN_API/blob/master/NN_v3.0_SandGlass/introduction/overview/README.md)

<img src="introduction/sandglass.png" width="200">

The multiple-layers attention mechanism is available for version > 3.02.
(see [introduction/attention_mechanism](https://github.com/hchungdelta/Simple_NN_API/tree/master/NN_v3.02_SandGlass/introduction/attention_mechanism))
<img src="https://github.com/hchungdelta/Simple_NN_API/blob/master/NN_v3.02_SandGlass/introduction/attention_mechanism/attention_mechanism.gif" width="500">
 
## environment:

* Python 3.5
* Numpy 1.14
* numba 0.39.0
* (optional) mpi4py

## .py file:
* **main.py**: training mode for reverse input test.
```
for example,
input = [1, 4, 5, 2, 0, 0]
target = [0, 0, 2, 5, 4, 1]
```
* **infer.py**: infer mode.
* **random_gene.py**: generate random lists with random lengths.

## Update:
#### layers: 
* 5 different kinds of convolutional layers (see [introduction/equation](https://github.com/hchungdelta/Simple_NN_API/tree/master/NN_v3.0_SandGlass/introduction/equation))

* attention mechanism (see [introduction/attention_mechanism](https://github.com/hchungdelta/Simple_NN_API/tree/master/NN_v3.02_SandGlass/introduction/attention_mechanism))

#### normalization:
BN(Batch Normalization)
#### training model:
[see TrainModel](https://github.com/hchungdelta/Simple_NN_API/tree/master/NN_v3.0_SandGlass/ML/TrainModel)

