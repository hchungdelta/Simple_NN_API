"""
Title: EndeModel
Description:
1. optimizer (SGD, momentum, adam)
2. parallel-computing.
3. save/restore.
"""
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
    def __init__(self, lr, mode='SGD', clipping=False, clip_value=1):
        # for MPI
        global comm, rank, size
        self.comm = comm
        self.size = size
        self.rank = rank

        self.lr = lr                  # learning rate
        self.mode = mode              # SGD, momentum or adam
        self.time = -1

        self.forward = []              # forward propagation, for both training and predicting.
        self.backward = []             # back propagation
        self.namelist = []             # store all the types of layers in the architecture.
        self.try_get_dWb = []          # the same size as namelist, if not trainable return (None)
        self.try_get_other = []        # get other parameter ...

        self.trainable_get_dWb = []
        self.trainable_update = []     # trainable classes only. For updating  W, b (e.g. W -> W-dW)
        self.rewrite_Wb = []           # trainable classes only. For replacing W, b (e.g. W -> newW)
        self.get_Wb = []               # trainable classes only. To get W,b from layers.
        self.dW = []                   # list of arrays (d_weight), for updating W
        self.db = []                   # list of arrays (d_bias)  , for updating b
        self.W = []                    # list of arrays (weight)
        self.b = []                    # list of arrays (bias)
        self.count_layer_number = 0
        self.descriptions = ''
        # clipping
        self.do_clipping = clipping
        self.clip_value = clip_value

    def add(self, layer):
        """
        add new layer to the model.
        """
        self.forward.append(layer.forward)
        self.backward.append(layer.backprop)
        self.count_layer_number += 1
        if hasattr(layer, 'get_dWb'):
            if self.rank == 0:
                this_W, this_b = layer.get_Wb()
                if isinstance(this_W, list):
                    for idx in range(len(this_W)):
                        this_W_shape = ','.join(['%4d'%_ for _ in this_W[idx].shape])
                        this_b_shape = ','.join(['%4d'%_ for _ in this_b[idx].shape])
                        W_shape = str('(') + this_W_shape + str(')')
                        b_shape = str('(') + this_b_shape + str(')')
                        if idx == 0:
                            self.print_layer_info(self.count_layer_number,
                                                  layer.__class__.__name__,
                                                  W_shape,
                                                  b_shape,
                                                  "trainable")
                        else:
                            self.print_layer_info(None, None, W_shape, b_shape, "trainable")

                else:
                    W_shape = ','.join(['%4d'%_ for _ in this_W.shape])
                    W_shape = str('(') + W_shape + str(')')
                    if isinstance(this_b, np.ndarray):
                        b_shape = ','.join(['%4d'%_ for _ in this_b.shape])
                        b_shape = str('(') + b_shape + str(')')
                    else:
                        b_shape = '   ---'
                    self.print_layer_info(self.count_layer_number,
                                          layer.__class__.__name__,
                                          W_shape,
                                          b_shape,
                                          "trainable")

            self.namelist.append(layer.__class__.__name__)
            self.trainable_update.append(layer.update)
            self.try_get_dWb.append(layer.get_dWb)
            self.get_Wb.append(layer.get_Wb)
            self.rewrite_Wb.append(layer.rewrite_Wb)
        else:
            if self.rank == 0:
                self.print_layer_info(self.count_layer_number,
                                      layer.__class__.__name__,
                                      '   ---',
                                      '   ---',
                                      "not trainable")
                if hasattr(layer, 'description'):
                    self.descriptions += "{:15s}: {:54s} ".format(layer.__class__.__name__,
                                                                  layer.description())+str('\n')
            self.namelist.append(layer.__class__.__name__)
            self.try_get_dWb.append(None)

        if hasattr(layer, 'get_parameter'):
            self.try_get_other.append(layer)
        if not hasattr(layer, 'get_parameter'):
            self.try_get_other.append(None)

    def print_layer_info(self, layer_idx, name, W_shape, b_shape, istrainable):
        """
        print out information.
        """
        if layer_idx != None:
            print("Layer {:3d}   {:15s}  W:{:28s} b:{:20s}     {}".format(layer_idx,
                                                                          name,
                                                                          W_shape,
                                                                          b_shape,
                                                                          istrainable))
        else:
            print(" {:8s}   {:17s}  {:30s} {:20s}     {}".format(" ",
                                                                 " ",
                                                                 W_shape,
                                                                 b_shape,
                                                                 istrainable))
    def show_detail(self):
        separator = str('=')*96 + str('\n')
        return separator + self.descriptions + separator

    def Forward(self, output_data, show=False, show_type="all"):
        output_dict = dict()
        for idx, func in enumerate(self.forward):
            output_data = func(output_data)
            this_idx = str('L')+str(idx+1)+str(':')
            if show == True and show_type == 'all':
                output_dict.update({this_idx+self.namelist[idx]:output_data})
            if show == True and show_type == 'absmean':
                output_dict.update({this_idx+self.namelist[idx]:float(
                    "%.3f"%np.mean(abs(output_data)))})

        return output_data if not show else (output_data, output_dict)

    def Backprop(self, dLoss, show=False, show_type="all"):
        dLoss_dict = dict()
        for idx, func in enumerate(reversed(self.backward)):
            dLoss = func(dLoss)
            this_idx = str('L')+str(len(self.backward)-idx)+str(':')
            if show == True and show_type == 'all':
                dLoss_dict.update({this_idx+self.namelist[len(self.backward)-(idx+1)]:dLoss})
            if show == True and show_type == 'absmean':
                dLoss_dict.update({this_idx+self.namelist[len(self.backward)-(idx+1)]:float(
                    "%.3f"%np.mean(abs(dLoss)))})

            if self.try_get_dWb[len(self.backward) - (idx+1)] != None:
                dW, db = self.try_get_dWb[len(self.backward) - (idx+1)]()
                self.dW.insert(0, dW)
                self.db.insert(0, db)
        # if no parallel computing
        if self.comm == None:
            self.sum_dW = self.dW
            self.sum_db = self.db
            self.sum_dW = self.destroy(self.sum_dW)
            self.sum_db = self.destroy(self.sum_db)
            self.Update()
            self.db = []
            self.dW = []
        # if yes :
        else:
            # master
            if self.rank == 0:
                self.sum_dW = self.destroy(self.dW)
                self.sum_db = self.destroy(self.db)

                for worker in range(1, self.size):
                    self.dW = self.comm.recv(source=worker)
                    self.db = self.comm.recv(source=worker)
                    self.dW = self.destroy(self.dW)
                    self.db = self.destroy(self.db)
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
        return  dLoss_dict if not show else None
    def Update_all(self):
        for get_dWb in self.try_get_dWb:
            if get_dWb != None:
                dW, db = get_dWb()
                self.dW.append(dW)
                self.db.append(db)
            # gather all dW/db from all processors
        # if no parallel computing
        if self.comm == None:
            self.sum_dW = self.dW
            self.sum_db = self.db
            self.Update()
            self.db = []
            self.dW = []
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
            self.time = 1

    def SGD(self):
        if self.do_clipping:
            for idx in range(len(self.sum_dW)):
                self.sum_dW[idx] = self.clipping(self.sum_dW[idx])
                self.sum_db[idx] = self.clipping(self.sum_db[idx])

        self.sum_dW = self.reconstruct(self.sum_dW)
        self.sum_db = self.reconstruct(self.sum_db)
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
            self.sum_adam_dW = []
            self.sum_adam_db = []
        for idx in range(len(self.sum_dW)):
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

                #update_func(adam_dW,adam_db,self.lr)
                self.sum_adam_dW.append(adam_dW)
                self.sum_adam_db.append(adam_db)

            if self.time == -1:
                # initialize and performance  first time update.
                self.sum_adam_dW.append(self.sum_dW[idx])
                self.sum_adam_db.append(self.sum_db[idx])
                #update_func(self.sum_dW[idx],self.sum_db[idx],self.lr)
                # previous step
                self.first_m_forW.append((1-self.beta_1)*self.sum_dW[idx])
                self.second_m_forW.append((1-self.beta_2)*(self.sum_dW[idx]**2))
                self.first_m_forb.append((1-self.beta_1)*self.sum_db[idx])
                self.second_m_forb.append((1-self.beta_2)*(self.sum_db[idx]**2))
            if self.do_clipping:
                self.sum_adam_dW[idx] = self.clipping(self.sum_adam_dW[idx])
                self.sum_adam_db[idx] = self.clipping(self.sum_adam_db[idx])

        self.sum_adam_dW = self.reconstruct(self.sum_adam_dW)
        self.sum_adam_db = self.reconstruct(self.sum_adam_db)
        self.time += 1
        if self.time == 0:
            self.time += 1
        for idx, update_func in enumerate(self.trainable_update):
            update_func(self.sum_adam_dW[idx], self.sum_adam_db[idx], self.lr)
        self.sum_adam_dW = []
        self.sum_adam_db = []


    def momentum(self):
        if self.do_clipping:
            for idx in range(len(self.sum_dW)):
                self.sum_dW[idx] = self.clipping(self.sum_dW[idx])
                self.sum_db[idx] = self.clipping(self.sum_db[idx])

        if self.time == -1:
            self.fraction = 0.9
        self.sum_dW = self.reconstruct(self.sum_dW)
        self.sum_db = self.reconstruct(self.sum_db)

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

    def clipping(self, inp):
        output = np.where(inp > self.clip_value, self.clip_value, inp)
        output = np.where(output < -self.clip_value, -self.clip_value, output)
        return output


    def Bcast_Wb(self, initial=False):
        """
        if initial= False : get W,b from rank 0
        else  : get W,b from rank 0, also to make placeholders for other processors
        """
        if initial:
            rank_need_to_init = self.size
        else:
            rank_need_to_init = 1
        # to get W and b.
        if self.rank < rank_need_to_init:
            for layer in self.get_Wb:
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
        """
        save trainable variables in savepath. (in .pickle)
        """
        if self.rank == 0:
            for layer in self.get_Wb:
                _W, _b = layer()
                self.W.append(_W)
                self.b.append(_b)
            other_parameter = []
            for layer in self.try_get_other:
                if layer != None:
                    other_parameter.append(layer.get_parameter())
                else:
                    other_parameter.append(None)
            trainable_vars = {"weights":self.W, "biases":self.b, "other_parameter":other_parameter}
            with open(savepath, 'wb') as pkfile:
                pickle.dump(trainable_vars, pkfile)
            print("saved!")
            print("save other variables in  :", savepath)
            # clear up
            self.W = []
            self.b = []
    def Restore(self, savepath):
        """
        restore trainable variables
        """
        print("restore trainable variables from :", savepath)
        with open(savepath, 'rb') as pkfile:
            trainable_vars = pickle.load(pkfile)

        self.W = trainable_vars["weights"]
        self.b = trainable_vars["biases"]
        other_parameter = trainable_vars["other_parameter"]
        for idx, rewrite_func in  enumerate(self.rewrite_Wb):
            rewrite_func(self.W[idx], self.b[idx])
        for idx, layer in enumerate(self.try_get_other):
            if layer != None:
                layer.rewrite_parameter(other_parameter[idx])
            else:
                pass
        # clear up
        self.W = []
        self.b = []

    def destroy(self, lista):
        """
        to break the sub-lists in list.
        e.g. [dW1,dW2,[dW3,dW4]]->[dW1,dW2,dW3,dW4]
        """
        return_list = []
        self.counter = []
        for item in lista:
            if isinstance(item, list):
                for sub_item in item:
                    return_list.append(sub_item)
                self.counter.append(len(item))
            else:
                return_list.append(item)
                self.counter.append(1)
        return return_list

    def reconstruct(self, lista):
        """
        reconstruct the list.
        e.g. [dW1,dW2,dW3,dW4]->[dW1,dW2,[dW3,dW4]]
        """
        recon_list = []
        offset = 0
        for count in self.counter:
            if count == 1:
                recon_list.append(lista[offset])
            else:
                recon_list.append([lista[offset:offset+count]])
            offset += count
        return recon_list
