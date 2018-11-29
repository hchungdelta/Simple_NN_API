from operator import add
import numpy as np
import _pickle as pickle

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except:
    print("cannot import MPI from mpi4py.")
    print("Hence only use single processor")
    comm = None
    rank = 0
    size = 1

class Model:
    def __init__(self, lr):
        # for MPI
        global comm, rank, size
        self.comm = comm
        self.size = size
        self.rank = rank

        self.lr = lr                # learning rate
        self.forward = []           # forward propagation, for both training and predicting.
        self.backward = []          # back propagation
        self.namelist = []          # store all the types of layers in the architecture.
        self.try_get_dWb = []       # the same size as namelist, if not trainable return (None)

        # below are lists for trainable classes only
        self.trainable_get_dWb = []
        self.trainable_update = []    # W -> W -dW
        self.rewrite_Wb = []          # W -> newW
        self.get_Wb = []              # get W and b
        self.dW = []
        self.db = []
        self.W = []
        self.b = []
        self.count_layer_number = 0

        self.mode = "SGD"
        self.time = -1

    def add(self, layer):
        self.forward.append(layer.forward)
        self.backward.append(layer.backprop)
        self.count_layer_number += 1
        if hasattr(layer, 'get_dWb'):
            if self.rank == 0:
                print("Layer {:3d}   {:15s}  trainable".format(self.count_layer_number,
                                                               layer.__class__.__name__))
            self.namelist.append(layer.__class__.__name__)
            self.trainable_update.append(layer.update)
            self.try_get_dWb.append(layer.get_dWb)
            self.get_Wb.append(layer.get_Wb)
            self.rewrite_Wb.append(layer.rewrite_Wb)
            #self.trainable_get_dWb(layer.get_dWb)
        else:
            if self.rank == 0:
                print("Layer {:3d}   {:15s}  not trainable".format(self.count_layer_number,
                                                                   layer.__class__.__name__))
            self.namelist.append(layer.__class__.__name__)
            self.try_get_dWb.append(None)
    def Forward(self, input_data):
        for func in self.forward:
            input_data = func(input_data)
        self.output = input_data
        return self.output
    def Backprop(self, dLoss):
        for idx, func in enumerate(reversed(self.backward)):
            dLoss = func(dLoss)
            if self.try_get_dWb[len(self.backward) - (idx+1)] != None:
                dW, db = self.try_get_dWb[len(self.backward) - (idx+1)]()
                self.dW.insert(0, dW)
                self.db.insert(0, db)
        # if no parallel computing
        if self.comm == None:
            self.Update()
        # if yes :
        else:
            # master
            if self.rank == 0:
                self.Update()
                for worker in range(1, self.size):
                    self.dW = self.comm.recv(source=worker)
                    self.db = self.comm.recv(source=worker)
                    self.Update()
            # workers
            if self.rank != 0:
                self.comm.send(self.dW, dest=0)
                self.comm.send(self.db, dest=0)
                # clear up, for memory concern.
                self.db = []
                self.dW = []
            self.Bcast_Wb()
    def Update_all(self, mode="SGD"):
        self.mode = mode
        for get_dWb in self.try_get_dWb:
            dW, db = get_dWb()
            self.dW.append(dW)
            self.db.append(db)
        # if no parallel computing
        if self.comm == None:
            self.sum_dW = self.dW
            self.sum_db = self.db
            self.Update()
            self.dW=[]
            self.db=[]
        # if yes :
        else:
            # master
            if self.rank == 0:
                self.sum_dW = self.dW
                self.sum_db = self.db
                for worker in range(1, self.size):
                    self.dW = self.comm.recv(source=worker)
                    self.db = self.comm.recv(source=worker)
                    self.sum_dW = list(map(add, self.sum_dW, self.dW))
                    self.sum_db = list(map(add, self.sum_db, self.db))
                self.Update()
            # workers
            if self.rank != 0:
                self.comm.send(self.dW, dest=0)
                self.comm.send(self.db, dest=0)
                # clear up, for memory concern.
            self.db = []
            self.dW = []
            self.Bcast_Wb()
    def Update(self):
        # W = W - dW,  b = b - db
        if self.mode == "SGD":
            self.SGD()
        if self.mode == "momentum":
            self.momentum()
        if self.mode == "adam":
            self.adam()
        if self.time == -1:
            self.time = 0
        self.time += 1
    def SGD(self):
        # W = W - dW, b = b -db
        for idx, update_func in enumerate(self.trainable_update):
            update_func(self.sum_dW[idx], self.sum_db[idx], self.lr)

    def adam(self):
        if self.time == -1:
            # initialize
            self.first_m_forW = []
            self.second_m_forW = []
            self.first_m_forb = []
            self.second_m_forb = []
            self.beta_1 = 0.9
            self.beta_2 = 0.999
            self.eps = 1e-8

        for idx, update_func in enumerate(self.trainable_update):
            if  self.time >= 0:
                self.first_m_forW[idx] = self.beta_1*self.first_m_forW[idx] +\
                (1-self.beta_1)*self.sum_dW[idx]

                self.second_m_forW[idx] = self.beta_2*self.second_m_forW[idx] +\
                (1-self.beta_2)*(self.sum_dW[idx]**2)

                first_m_forW_ = self.first_m_forW[idx]/(1-self.beta_1**self.time)
                second_m_forW_ = self.second_m_forW[idx]/(1-self.beta_2**self.time)

                adam_dW = first_m_forW_/(np.sqrt(second_m_forW_)+self.eps)

                self.first_m_forb[idx] = self.beta_1*self.first_m_forb[idx] +\
                (1-self.beta_1)*self.sum_db[idx]

                self.second_m_forb[idx] = self.beta_2*self.second_m_forb[idx] +\
                (1-self.beta_2)*(self.sum_db[idx]**2)

                first_m_forb_ = self.first_m_forb[idx]/(1-self.beta_1**self.time)
                second_m_forb_ = self.second_m_forb[idx]/(1-self.beta_2**self.time)

                adam_db = first_m_forb_/(np.sqrt(second_m_forb_)+self.eps)

                update_func(adam_dW, adam_db, self.lr)
            if self.time == -1:
                # initialize and performance  first time update.
                update_func(self.sum_dW[idx], self.sum_db[idx], self.lr)
                # previous step
                self.first_m_forW.append((1-self.beta_1)*self.sum_dW[idx])
                self.second_m_forW.append((1-self.beta_2)*(self.sum_dW[idx]**2))
                self.first_m_forb.append((1-self.beta_1)*self.sum_db[idx])
                self.second_m_forb.append((1-self.beta_2)*(self.sum_db[idx]**2))
    def momentum(self):
        if self.time == -1:
            # initialize
            self.fraction = 0.9

        for idx, update_func in enumerate(self.trainable_update):
            update_func(self.sum_dW[idx], self.sum_db[idx], self.lr)
            if  self.time >= 0:
                update_func(self.dW_prev[idx], self.db_prev[idx], self.lr*self.fraction)

        # clean up
        self.dW_prev = []
        self.db_prev = []
        # remember previous step displacements (for momentum)
        for idx, update_func in enumerate(self.trainable_update):
            # record previous step
            self.dW_prev.append(self.sum_dW[idx])
            self.db_prev.append(self.sum_db[idx])

    def Bcast_Wb(self, initial=False):
        # if initial= False : get W, b from rank 0
        # else  : get W,b from rank 0, also to make placeholders for other processors
        if initial:
            rank_need_to_init = self.size
        else:
            rank_need_to_init = 1

        # to get W and b.
        if self.rank < rank_need_to_init:
            for layer in  self.get_Wb:
                _W, _b = layer()
                self.W.append(_W)
                self.b.append(_b)
        # bcast the W and b in rank 0 to all the other workers
        self.W = self.comm.bcast(self.W, root=0)
        self.b = self.comm.bcast(self.b, root=0)
        # after receving self.W and self.b from rank 0, replace the original ones.
        for idx, rewrite_func in enumerate(self.rewrite_Wb):
            rewrite_func(self.W[idx], self.b[idx])
        # clean up.
        self.W = []
        self.b = []
    def Save(self, savepath):
        # save trainable variables.
        if self.rank == 0:
            for layer in  self.get_Wb:
                _W, _b = layer()
                self.W.append(_W)
                self.b.append(_b)
            trainable_vars = {"weights":self.W, "biases":self.b}
            print("save trainable variables in  :", savepath)
            with open(savepath, 'wb') as pkfile:
                pickle.dump(trainable_vars, pkfile)
            print("saved!")

            # clear up
            self.W = []
            self.b = []
    def Restore(self, savepath):
        # restore trainable variables
        print("restore trainable variables from :", savepath)
        with open(savepath, 'rb') as pkfile:
            trainable_vars = pickle.load(pkfile)

        self.W = trainable_vars["weights"]
        self.b = trainable_vars["biases"]
        for idx, rewrite_func in enumerate(self.rewrite_Wb):
            rewrite_func(self.W[idx], self.b[idx])
        # clear up
        self.W = []
        self.b = []
