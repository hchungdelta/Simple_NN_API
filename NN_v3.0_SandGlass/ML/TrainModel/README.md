# Training Model
EndeModel is short for the Encoder/Decoder Model.
This model determines how to update the weights and bias in each layer.

The functions of this model:

#### 1. How to update the weights and bias in each layer.
* learning rate
* optimizer : support SGD, momentum, and adam.
```
#example
EndeModel = ML.EndeModel.Model(lr=0.0012) #lr is short for learning rate.
EndeModel.add(Bi_Encoder1) #register the trainable layer.
EndeModel.add(Decoder1)
EndeModel.add(linearlayer)

...(coding)...

EndeModel.Update_all(mode="adam") # adopt adam optimizer
```

#### 2. Save and restore in pickle
```
# usage
EndeModel.Save('data_path.pickle')
EndeModel.Restore('data_path.pickle')
```

#### 3. Embarrassing parallel computing

If parallel computing is available, the model will broadcast the same initial weights/bias to all processors at the beginning.
```
# initializer
if EndeModel.comm != None :
    EndeModel.Bcast_Wb(initial=True)
```
And also, one must make sure the training data are separated to each processor correctly, as  in ```seq2seq.py```.
At the end of each training step, ask the model to update all parameters.

```
EndeModel.Update_all(mode="adam") # adopt adam optimizer
```

The following illustrates the process of training with parallel computing.
<img src="parallel_comput.gif" width="800">


