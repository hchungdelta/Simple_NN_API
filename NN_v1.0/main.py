import numpy as np
import time
import MNIST_data_loader as MNIST_load
import ML  
comm = ML.TrainModel.comm
rank = ML.TrainModel.rank
size = ML.TrainModel.size
'''
mpi4py:
comm = parallel computing commander  /None if MPI not exist
rank = index of this processor       /0    if MPI not exist 
size = amount of processors          /1    if MPI not exist
'''

# Import data
training_data, validation_data, test_data = MNIST_load.load_data()

# record the cost of time
start = time.time()

# save path for trainable variables 
savepath = 'data/trainable_vars.pickle'

# Build up neural network architecture for all processors  
# note that the initial weights/bias are different for each processor
Training_Model = ML.TrainModel.Model(lr=0.100)
Training_Model.add(ML.Layer.CNN_layer.CNN_layer((2, 1, 28, 28),paddling=True , kernel_size=(2,2),stride=(1,1),activation="None"))
Training_Model.add(ML.Layer.Acti_layer.Tanh(upperlimit=1,smooth=10))
Training_Model.add(ML.Layer.CNN_layer.max_pooling())
Training_Model.add(ML.Layer.CNN_layer.CNN_layer((4, 2, 14, 14),paddling=True, kernel_size=(2,2),stride=(1,1),activation="None"))
Training_Model.add(ML.Layer.Acti_layer.Tanh(upperlimit=1,smooth=10))
Training_Model.add(ML.Layer.CNN_layer.max_pooling())
Training_Model.add(ML.Layer.CNN_layer.flatten())
Training_Model.add(ML.Layer.FCL_layer.xW_b([4*7*7,49]))
Training_Model.add(ML.Layer.Acti_layer.Tanh())
Training_Model.add(ML.Layer.FCL_layer.xW_b([49,10]))

# bcast the initial weights and bias from rank 0 to all the others


if Training_Model.comm != None :
    pass
    #Training_Model.Bcast_Wb(initial=True)

Training_Model.Restore('data/trainable_vars.pickle')

if rank >= 0 :
    # Every processors share the same initial weights and bias
    batch = 32
    training_data_amount = len(training_data)

    total_step_in_one_loop = int(training_data_amount/(batch*size) )
    display_epoch = int(  0.1*total_step_in_one_loop   )
    amount_of_loop = 12

    # check accuracy with validation data
    use_validation = True   #(validation data)
    display_epoch_for_validation = total_step_in_one_loop

    
    total_L=0.
    offset=0

    for a in range(amount_of_loop*total_step_in_one_loop+1):
        # use validation data to estimate the accuracy
        if  a %  display_epoch_for_validation ==0  and a !=0 and use_validation == True:
            validation_target=[]
            validation_batch =[]
            # separate the job to N processors 
            validation_data_batchsize= int(   np.array(validation_data).shape[0]/size)
            for single_data in range(validation_data_batchsize*rank,validation_data_batchsize*(rank+1)  ) :
                validation_batch.append(  validation_data[single_data][0]  )
                validation_target.append( validation_data[single_data][1]  )

            # arrayize
            validation_batch = np.array(validation_batch)
            validation_target= np.array(validation_target)

            output = Training_Model.Forward(validation_batch.reshape(-1,1,28,28))
            pred , _ , _  = ML.NN.Loss.softmax_cross_entropy(output,validation_target)

            TARGET= np.argmax(validation_target,axis=1 )
            PRED  = np.argmax(pred,axis=1 )
            accuracy=ML.NN.Tools.accuracy_test(  TARGET , PRED)
            gather_accuracy = comm.gather( accuracy,root=0)
            if rank == 0 :
                print("Accuracy of validation data : {:.3f}".format(sum(gather_accuracy)/size)  )

        # generate input / real output
        target=[]
        training_batch=[]
        if offset+batch*(rank+1) > training_data_amount :
            offset=0  # note that although simple, this method will abandon the last few data.
        for single_data in range(offset+batch*rank,offset+batch*(rank+1)  ) :
            training_batch.append(training_data[single_data][0])  
            target.append(training_data[single_data][1])
         
        offset += batch*size
        # arrayize 
        training_batch=np.array(training_batch)
        target= np.array(target)
        # forward
        output = Training_Model.Forward(training_batch.reshape(-1,1,28,28))
        # softmax criss entropy / loss function
        pred , L , dLoss  = ML.NN.Loss.softmax_cross_entropy(output,target)
        # Loss by average
        total_L += L/display_epoch
        # back propagation
        Training_Model.Backprop(dLoss)
           
        # use training data to estimate the accuracy
        if a %  display_epoch==0  and rank ==0 and a != 0 : 
            PRED  =np.argmax(pred,axis=1 )
            TARGET=np.argmax(target,axis=1 )
            print("step",a,"Loss", total_L)
            print("target: {}".format(TARGET)  ) 
            print("output: {}".format(PRED  )  )
            accuracy=ML.NN.Tools.accuracy_test( TARGET , PRED )
            print("Accuracy of trainin data : {:.3f}".format(accuracy)  )
            total_L=0
            Training_Model.Save(savepath)
    if rank ==0 :
        print("COST:{:.3f} seconds".format(time.time() - start) )

