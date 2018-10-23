# Simple_NN_API
Personal neural network API from scratch. Only using numpy.

Example code:
```
Training_Model=ML.TrainModel.Model(lr=0.100)
Training_Model.add(ML.Layer.CNN_layer.CNN_layer( ( 2,1,28,28),paddling=True , kernel_size=(2,2),stride=(1,1),activation="None"))
Training_Model.add(ML.Layer.Acti_layer.Tanh(upperlimit=1,smooth=10))
Training_Model.add(ML.Layer.CNN_layer.max_pooling())
Training_Model.add(ML.Layer.CNN_layer.CNN_layer( ( 4,2,14,14),paddling=True, kernel_size=(2,2),stride=(1,1),activation="None"))
Training_Model.add(ML.Layer.Acti_layer.Tanh(upperlimit=1,smooth=10))
Training_Model.add(ML.Layer.CNN_layer.max_pooling())
Training_Model.add(ML.Layer.CNN_layer.flatten())
Training_Model.add(ML.Layer.FCL_layer.xW_b([4*7*7,49]))
Training_Model.add(ML.Layer.Acti_layer.Tanh())
Training_Model.add(ML.Layer.FCL_layer.xW_b([49,10]))
```
 


# Environment :
- Python 3.X
- Numpy 1.14.3
- (optional) mpi4py


# Descriptions

## v1.0

- main.py : The place to build neural network structure.
- mnist_data_loader.py : to load MNIST handwritten digits.

- (file) ML : Neural network structure.

Currently support:

#### Layer:

1.Convlutional layers (CNN) / maxpooling 

2.Fully-connected layers.

3.Activation : Linear (default), sigmoid, ReLU, tanh.

#### others:

3.Dropout mechanisms


## parallel computing

"batchsize"-wise parallel computing is supported in this code. (Using mpi4py) 

In MNIST example, TrainModel.py (in ML/TrainModel/TrainModel.py) will try to import mpi4py and performs parallel computing if available.  

MNIST_run : to submit computing job. 
or just :
mpirun -np (amount_of_ppn) yourjob.py    

## save/restore

 Data can be saved/restored using "pickle"

```
#Save
savepath = 'data/trainable_vars.pickle'
Training_Model=ML.TrainModel.Model(lr=0.100)  
......(some code)......
Training_Model.Save(savepath)
```


```
#Restore
savepath = 'data/trainable_vars.pickle'
Training_Model.Restore(savepath)
```
