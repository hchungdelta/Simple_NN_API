# SandGlass Conv2Conv model

The basic building block is the following equation. 

<img src="equation_graph/conv_equation.gif" width="280">

"x" is the inputs (embedding or previous convolutional layer). It gathers the information from previous layers. Hence each output possesses the information of the kernel size. The more the convolutional layers, the larger the information it can gain.

As in CNN, the amount of filter can be more than one, and could have different kernel sizes.
Therefore, in general, the equation can be expressed as:

<img src="equation_graph/conv_equation2.gif" width="450">


Based on these ideas, I have further developed three other types of convolutional layers, including reverse convolutional layer (reverse conv.), reduce convolutional layer (reduce conv.), and expand convolutional layer (expand conv.) As what follows:

<img src="equation_graph/conv_equation3.gif" width="520">


<img src="equation_graph/conv.gif" width="620">
<img src="equation_graph/reverseconv.gif" width="620">
<img src="equation_graph/reduceconv.gif" width="620">
<img src="equation_graph/expandconv.gif" width="620">

## others:

* Reduce Attention (testing):
