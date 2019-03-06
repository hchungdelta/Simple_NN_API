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
    def __init__(self, lr, optimizer='SGD', mode='train', clipping=False, clip_value=1):
        # for MPI
        global comm, rank, size
        self.comm = comm
        self.size = size
        self.rank = rank

        self.mode = mode             # train(teacher_force), or infer
        self.optimizer = optimizer   # SGD, momentum or adam
        self.lr = lr                 # learning rate
        self.time = -1
        self.try_get_dWb = []
        self.namelist = []             # store all the types of layers in the architecture.
        self.try_get_other = []        # get other parameter ...

        self.trainable_get_dWb = []
        self.trainable_update = []     # trainable classes only. For updating  W, b (e.g. W -> W-dW)
        self.rewrite_Wb = []           # trainable classes only. For replacing W, b (e.g. W -> newW)
        self.get_Wb = []               # trainable classes only. To get W,b from layers.
        self.dW = []                   # list of arrays (d_weight), for updating W
        self.db = []                   # list of arrays (d_bias)  , for updating b
        self.W = []                    # list of arrays (weight)
        self.b = []                    # list of arrays (bias)

        # encoder layers / decoder layers
        self.encoder_forward = []
        self.encoder_backprop = []
        self.encoder_get_dWb = []
        self.encoder_connection_label = []
        self.encoder_isLSTM = []

        self.decoder_forward = []
        self.decoder_backprop = []
        self.decoder_get_dWb = []
        self.decoder_connection_label = []
        self.decoder_isLSTM = []  # is LSTM cell? (True/False)
        self.decoder_isCost = []  # is cost function? (True/False)
        self.decoder_isConcat = []
        self.info_repository = dict() # information delivered between encoder and decoder
        self.cutoff_length = None
        self.encoder_layer_number = 0
        self.decoder_layer_number = 0

        self.try_timestep_gather = []  # to clear up the memory (for infer mode)
        self.descriptions = ''
        # clipping
        self.do_clipping = clipping
        self.clip_value = clip_value

    def add(self, layer, belong_to, just_forward=False, connection_label=None):
        """
        add new layer to the model.
        belong_to: Encoder or Decoder
        just_forward: similiar to forward function, except this function doesn't remember the input
                      and cannot be used to train the weights and bias in the layer.
        *developing note: if just_forward appear in the mid of forward functions,
                          just_backprop is necessary.
        """
        if belong_to == "Encoder":
            self.encoder_layer_number += 1
            count_layer_number = self.encoder_layer_number
            self.encoder_connection_label.append(connection_label)
            if layer.__class__.__name__ in ["BiLSTM", "LSTMcell"]:
                self.encoder_isLSTM.append(True)
            else:
                self.encoder_isLSTM.append(False)
            if not just_forward:
                self.encoder_forward.append(layer.forward)
                self.encoder_backprop.append(layer.backprop)
                istrainable = "trainable"
            if just_forward:
                self.encoder_forward.append(layer.just_forward)
                self.encoder_backprop.append(layer.just_backprop)
                istrainable = "not trainable"

        if belong_to == "Decoder":
            if not just_forward and self.mode == 'train':
                forward_function = layer.forward
            if just_forward and self.mode == 'train':
                forward_function = layer.just_forward

            if not just_forward and self.mode == 'infer':
                forward_function = layer.timestep_forward
            if  just_forward and self.mode == 'infer':
                forward_function = layer.timestep_just_forward

            self.decoder_layer_number += 1
            count_layer_number = self.decoder_layer_number
            self.decoder_connection_label.append(connection_label)
            if layer.__class__.__name__ in ["BiLSTM", "LSTMcell"]:
                self.decoder_isLSTM.append(True)
            else:
                self.decoder_isLSTM.append(False)

            if layer.__class__.__name__ in ["softmax_cross_entropy"]:
                self.decoder_isCost.append(True)
            else:
                self.decoder_isCost.append(False)
            if layer.__class__.__name__ in ["persona_concat"]:
                self.decoder_isConcat.append(True)
            else:
                self.decoder_isConcat.append(False)


            self.decoder_forward.append(forward_function)
            if not just_forward:
                self.decoder_backprop.append(layer.backprop)
                istrainable = "trainable"
            if just_forward:
                self.decoder_backprop.append(layer.just_backprop)
                istrainable = "not trainable"

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
                            self.print_layer_info(belong_to,
                                                  count_layer_number,
                                                  layer.__class__.__name__,
                                                  W_shape,
                                                  b_shape,
                                                  istrainable)
                        else:
                            self.print_layer_info(belong_to, None, None,
                                                  W_shape, b_shape, istrainable)

                else:
                    W_shape = ','.join(['%4d'%_ for _ in this_W.shape])
                    W_shape = str('(') + W_shape + str(')')
                    if isinstance(this_b, np.ndarray):
                        b_shape = ','.join(['%4d'%_ for _ in this_b.shape])
                        b_shape = str('(') + b_shape + str(')')
                    else:
                        b_shape = '   ---'
                    self.print_layer_info(belong_to,
                                          count_layer_number,
                                          layer.__class__.__name__,
                                          W_shape,
                                          b_shape,
                                          istrainable)

            self.namelist.append(layer.__class__.__name__)
            if not just_forward:
                self.trainable_update.append(layer.update)
                self.try_get_dWb.append(layer.get_dWb)
                self.get_Wb.append(layer.get_Wb)
                self.rewrite_Wb.append(layer.rewrite_Wb)
            if just_forward:
                self.try_get_dWb.append(None)

        else:
            if self.rank == 0:
                self.print_layer_info(belong_to,
                                      count_layer_number,
                                      layer.__class__.__name__,
                                      '   ---',
                                      '   ---',
                                      "not trainable")
                if hasattr(layer, 'description'):
                    self.descriptions += "{:25s}: {:54s} ".format(layer.__class__.__name__,
                                                                  layer.description())+str('\n')
            self.namelist.append(layer.__class__.__name__)
            self.try_get_dWb.append(None)

        if hasattr(layer, 'get_parameter'):
            self.try_get_other.append(layer)
        if not hasattr(layer, 'get_parameter'):
            self.try_get_other.append(None)
        if hasattr(layer, 'timestep_gather'):
            self.try_timestep_gather.append(layer)
        if not hasattr(layer, 'timestep_gather'):
            self.try_timestep_gather.append(None)



    def print_layer_info(self, belong_to, layer_idx, name, W_shape, b_shape, istrainable):
        """
        print out information.
        """
        if layer_idx != None:
            print("{} layer {:3d}   {:25s}  W:{:20s} b:{:15s}     {}".format(belong_to,
                                                                             layer_idx,
                                                                             name,
                                                                             W_shape,
                                                                             b_shape,
                                                                             istrainable))
        else:
            print(" {:8s}   {:40s}  {:17s} {:15s}     {}".format(" ",
                                                                 " ",
                                                                 W_shape,
                                                                 b_shape,
                                                                 istrainable))
    def show_detail(self):
        separator = str('=')*96 + str('\n')
        return separator + self.descriptions + separator

    def Forward(self, encoder_output_data, decoder_output_data, target,
                cutoff_length, persona_embedding=None, show=False, show_type="all"):
        """
        cutoff_length = [en_cutoff_length, de_cutoff_length]
        persona_embedding = concatenate decoder_output_data with  persona_embedding
        """
        self.cutoff_length = cutoff_length
        output_dict = dict()
        for idx, func in enumerate(self.encoder_forward):
            if self.encoder_isLSTM[idx]:
                encoder_output_data, _, final_h, final_c = func(encoder_output_data,
                                                                None,
                                                                None,
                                                                cutoff_length=cutoff_length[0])
                if self.encoder_connection_label[idx] != None:
                    self.info_repository.update(
                        {self.encoder_connection_label[idx]:[final_h, final_c]})
            else:
                encoder_output_data = func(encoder_output_data)

            this_idx = str('L')+str(idx+1)+str(':')
            if show == True and show_type == 'all':
                output_dict.update({this_idx+self.namelist[idx]:encoder_output_data})
            if show == True and show_type == 'absmean':
                output_dict.update({this_idx+self.namelist[idx]:float(
                    "%.3f"%np.mean(abs(encoder_output_data)))})
            if show == True and show_type == 'max':
                output_dict.update({this_idx+self.namelist[idx]:float(
                    "%.3f"%np.max(abs(encoder_output_data)))})

        if self.mode == 'infer':
            temp_collection = []
            timestep_output = decoder_output_data
            Loss = 0
            for timestep in range(cutoff_length[1]):
                timestep_target = target[timestep] if isinstance(target, np.ndarray) else None
                for idx, func in enumerate(self.decoder_forward):
                    this_forward_done = False
                    if self.decoder_isLSTM[idx]:
                        this_forward_done = True
                        if self.decoder_connection_label[idx] != None:
                            en_final_h, en_final_c = self.info_repository[self.decoder_connection_label[idx]]
                        else:
                            en_final_h, en_final_c = None, None
                        timestep_output, _, final_h, final_c = func(timestep_output,
                                                                    en_final_h,
                                                                    en_final_c,
                                                                    cutoff_length=cutoff_length[1])
                        self.info_repository.update({self.decoder_connection_label[idx]:[final_h, final_c]})
                    if self.decoder_isCost[idx]:
                        this_forward_done = True
                        softmax_output, _Loss = func(timestep_output, timestep_target)
                        Loss += _Loss
                        argmax_list = np.argmax(softmax_output, axis=1)
                        timestep_output = np.zeros_like(softmax_output)
                        for this_batch, _argmax in enumerate(argmax_list):
                            timestep_output[this_batch][_argmax] = 1
                    if self.decoder_isConcat[idx]:
                        this_forward_done = True
                        timestep_output = func(timestep_output, persona_embedding)
                    if not this_forward_done:
                        timestep_output = func(timestep_output)


                temp_collection.append(softmax_output)
            prediction = np.array(temp_collection)

        if self.mode == 'train':
            for idx, func in enumerate(self.decoder_forward):
                this_forward_done = False
                if self.decoder_isLSTM[idx]:
                    this_forward_done = True
                    if self.decoder_connection_label[idx] != None:
                        en_final_h, en_final_c = self.info_repository[self.decoder_connection_label[idx]]
                    else:
                        en_final_h, en_final_c = None, None
                    decoder_output_data, _, _, _ = func(decoder_output_data,
                                                        en_final_h,
                                                        en_final_c,
                                                        cutoff_length=cutoff_length[1])

                if self.decoder_isCost[idx]:
                    this_forward_done = True
                    prediction, Loss = func(decoder_output_data, target)
                if self.decoder_isConcat[idx]:
                    this_forward_done = True
                    decoder_output_data = func(decoder_output_data, persona_embedding)
                if not this_forward_done:
                    decoder_output_data = func(decoder_output_data)

                this_idx = str('L')+str(idx+1)+str(':')
                if show == True and show_type == 'all':
                    output_dict.update({this_idx+self.namelist[idx]:decoder_output_data})
                if show == True and show_type == 'absmean':
                    output_dict.update({this_idx+self.namelist[idx]:float(
                        "%.3f"%np.mean(abs(decoder_output_data)))})
                if show == True and show_type == 'max':
                    output_dict.update({this_idx+self.namelist[idx]:float(
                        "%.3f"%np.max(abs(decoder_output_data)))})
        return (prediction, Loss) if not show else (prediction, Loss, output_dict)

    def Backprop(self, show=False, show_type="all"):
        dLoss = None
        dLoss_dict = dict()
        for idx, func in enumerate(reversed(self.decoder_backprop)):
            this_backprop_done = False
            rev_idx = len(self.decoder_backprop)-(idx+1)
            global_rev_idx = len(self.try_get_dWb) - (idx+1)
            if self.decoder_isLSTM[rev_idx]:
                dLoss, dh, dc = func(dLoss, None, None, cutoff_length=self.cutoff_length[1])
                if self.decoder_connection_label[rev_idx] != None:
                    self.info_repository.update({self.decoder_connection_label[rev_idx]:[dh, dc]})
                this_backprop_done = True
            if self.decoder_isCost[rev_idx]:
                dLoss = func()
                this_backprop_done = True
            if self.decoder_isConcat[rev_idx]:
                dLoss, _ = func(dLoss)
                this_backprop_done = True
            if not this_backprop_done:
                dLoss = func(dLoss)

            this_idx = str('L')+str(len(self.decoder_backprop)-idx)+str(':')
            if show == True and show_type == 'all':
                dLoss_dict.update({this_idx+self.namelist[global_rev_idx]:dLoss})
            if show == True and show_type == 'absmean':
                dLoss_dict.update({this_idx+self.namelist[global_rev_idx]:float(
                    "%.3f"%np.mean(abs(dLoss)))})
            if show == True and show_type == 'max':
                dLoss_dict.update({this_idx+self.namelist[global_rev_idx]:float(
                    "%.3f"%np.max(abs(dLoss)))})

            if self.try_get_dWb[global_rev_idx] != None:
                dW, db = self.try_get_dWb[global_rev_idx]()
                self.dW.insert(0, dW)
                self.db.insert(0, db)
        for idx, func in enumerate(reversed(self.encoder_backprop)):
            rev_idx = len(self.encoder_backprop)-(idx+1)
            if self.encoder_isLSTM[rev_idx]:
                if self.encoder_connection_label[rev_idx] != None:
                    dh, dc = self.info_repository[self.encoder_connection_label[rev_idx]]
                else:
                    dh, dc = None, None
                dLoss, _, _ = func(dLoss, dh, dc, cutoff_length=self.cutoff_length[0])
            else:
                dLoss = func(dLoss)

            this_idx = str('L')+str(len(self.encoder_backprop)-idx)+str(':')
            if show == True and show_type == 'all':
                dLoss_dict.update({this_idx+self.namelist[rev_idx]:dLoss})
            if show == True and show_type == 'absmean':
                dLoss_dict.update({this_idx+self.namelist[rev_idx]:float(
                    "%.3f"%np.mean(abs(dLoss)))})
            if show == True and show_type == 'max':
                dLoss_dict.update({this_idx+self.namelist[rev_idx]:float(
                    "%.3f"%np.max(abs(dLoss)))})

            if self.try_get_dWb[rev_idx] != None:
                dW, db = self.try_get_dWb[rev_idx]()
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
        return dLoss if not show else (dLoss, dLoss_dict)
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
    def Update(self):
        # W = W - dW,  b = b - db
        if self.optimizer == "SGD":
            self.SGD()
        if self.optimizer == "momentum":
            self.momentum()
        if self.optimizer == "adam":
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

                self.sum_adam_dW.append(adam_dW)
                self.sum_adam_db.append(adam_db)


            if self.time == -1:
                # initialize and performance  first time update.
                self.sum_adam_dW.append(self.sum_dW[idx])
                self.sum_adam_db.append(self.sum_db[idx])
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


    def Timestep_gather(self):
        for func in self.try_timestep_gather:
            if func != None:
                func.timestep_gather()

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

        recon_list = []
        offset = 0
        for count in self.counter:
            if count == 1:
                recon_list.append(lista[offset])
            else:
                recon_list.append([lista[offset:offset+count]])
            offset += count
        return recon_list
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

