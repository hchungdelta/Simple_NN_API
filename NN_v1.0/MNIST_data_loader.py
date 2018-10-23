import _pickle as pickle
import gzip
import numpy as np

def load_data():
    file = gzip.open('MNIST/mnist.pkl.gz', 'rb')
    training_data,validation_data,test_data=pickle.load(file,encoding='latin1')
    file.close()
  
    training_results  = [onehot(y,10) for y in training_data[1]     ]
    training_data     = list( zip(training_data[0],training_results)      )
    validation_results= [onehot(y,10) for y in validation_data[1]   ]
    validation_data   = list( zip(validation_data[0],validation_results)  )
    test_results      = [onehot(y,10) for y in test_data[1]         ]
    test_data         = list( zip(test_data[0], test_results)             )
    return (training_data, validation_data, test_data)
 
def onehot(number,depth):
    array=np.zeros(depth,dtype='f')
    array[number]=1.
    return array
 
