# Simple_NN_API V 1.0
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

check accuracy using validation data when finishing one loop.

```
Accuracy of validation data : 0.808
Accuracy of validation data : 0.851
Accuracy of validation data : 0.873
Accuracy of validation data : 0.893
Accuracy of validation data : 0.904
Accuracy of validation data : 0.925
Accuracy of validation data : 0.930
Accuracy of validation data : 0.930
Accuracy of validation data : 0.946
Accuracy of validation data : 0.952
```

check accuracy using training data during training.

```
target: [0 9 3 0 3 1 5 2 5 3 0 4 3 5 5 6 8 7 3 8 6 9 1 0 7 5 1 3 1 9 5 0]
output: [0 9 3 0 3 1 5 2 5 3 0 4 3 5 5 6 8 7 8 8 6 9 1 0 7 5 1 3 1 9 5 0]
Accuracy of trainin data : 0.969
```



# Environment :
- Python 3.X
- Numpy 1.14.3
- (optional) mpi4py


# Descriptions

## MNIST example

- main.py : The place to build neural network structure.
- mnist_data_loader.py : to load MNIST handwritten digits.
- MNIST_run : to submit computing job. 
- (file) ML : Neural network structure.

## Currently support:

#### Layer:

1.Convlutional layers (CNN) / maxpooling 

2.Fully-connected layers.

3.Activation : Linear (default), sigmoid, ReLU, tanh.

#### others:

Dropout mechanisms


## parallel computing

"batchsize"-wise parallel computing is supported in this code. (Using mpi4py) 

In MNIST example, TrainModel.py (in ML/TrainModel/TrainModel.py) will try to import mpi4py and performs parallel computing if available. 

One can use :
```
mpirun -np (amount_of_ppn) yourjob.py    
```

Or using PBS as shown in MNIST_run``` MNIST_run ```


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
