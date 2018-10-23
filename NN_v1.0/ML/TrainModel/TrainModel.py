import _pickle as pickle
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except :
    print("cannot import MPI from mpi4py.")
    print("Hence only use single processor")
    comm=None
    rank=0
    size=1

class Model:
    def __init__(self,lr):
        # for MPI
        global comm , rank , size
        self.comm = comm
        self.size = size
        self.rank = rank
        
        self.lr  = lr          # learning rate
        self.forward=[]        # forward propagation, for both training and predicting.
        self.backward=[]       # back propagation
        self.namelist=[]       # store all the types of layers in the architecture.
        self.try_get_dWb=[]    # the same size as namelist, if not trainable return (None)
       
        self.trainable_update=[]     # list of arguments (trainable classes only). used for updating  W, b (e.g. W --> W - dW) 
        self.rewrite_Wb=[]           # list of arguments (trainable classes only). used for replacing W, b (e.g. W --> newW) 
        self.get_Wb =[]              # list of arguments (trainable classes only). to get W,b from layers.
        self.dW=[]                   # list of arrays (d_weight), for updating W
        self.db=[]                   # list of arrays (d_bias)  , for updating b
        self.W=[]                    # list of arrays (weight)  , for initialization and save/restore
        self.b=[]                    # list of arrays (bias)    , for initialization and save/restore

        self.count_layer_number=0

    def add(self,layer):
        self.forward.append(layer.forward)
        self.backward.append(layer.backprop)
        self.count_layer_number +=1
        if hasattr(layer, 'get_dWb'):
            if self.rank == 0 :
                print("Layer {:3d}   {:15s}  trainable".format(self.count_layer_number,layer.__class__.__name__))
            self.namelist.append(layer.__class__.__name__)
            self.trainable_update.append(layer.update)
            self.try_get_dWb.append(layer.get_dWb)
            self.get_Wb.append(layer.get_Wb)
            self.rewrite_Wb.append(layer.rewrite_Wb)
        else :
            if self.rank ==0 :
                print("Layer {:3d}   {:15s}  not trainable".format(self.count_layer_number,layer.__class__.__name__))
            self.namelist.append(layer.__class__.__name__)
            self.try_get_dWb.append(None)
    def Forward(self, input_data):
        for func in self.forward :
            input_data=func(input_data)
        self.output = input_data
        return self.output
    def Backprop(self, dLoss):
        for idx,func in enumerate(reversed(self.backward)) :
            dLoss=func(dLoss)
            if self.try_get_dWb[len(self.backward) - (idx+1)] != None :
                dW,db=self.try_get_dWb[len(self.backward) - (idx+1)]()
                self.dW.insert(0,dW)
                self.db.insert(0,db)
        # if no parallel computing
        if self.comm == None:
            self.Update()
        # if yes :
        else :
            # master 
            if self.rank == 0 :
                self.Update()
                for worker in range(1,self.size):
                    self.dW = self.comm.recv(source=worker) 
                    self.db = self.comm.recv(source=worker)
                    self.Update()
            # workers
            if self.rank != 0 :
                self.comm.send(self.dW,dest=0)
                self.comm.send(self.db,dest=0)
                # clear up, for memory concern.
                self.db =[]
                self.dW= []
            self.Bcast_Wb()

    def Update(self):
        # W = W - dW,  b = b - db
        for idx,update_func in enumerate (self.trainable_update) :
            update_func(self.dW[idx],self.db[idx],self.lr)
    def Bcast_Wb(self,initial= False):
        # if initial= False : get W,b from rank 0
        # else  : get W,b from rank 0, also to make placeholders for other processors
        if initial : rank_need_to_init = self.size
        else :       rank_need_to_init = 1 
                
        # to get W and b.
        if self.rank  < rank_need_to_init :
            for layer in  self.get_Wb :
                _W,_b = layer()
                self.W.append(_W)
                self.b.append(_b)
        if rank ==0 : print("W \n", _W[0][0])
        # bcast the W and b in rank 0 to all the other workers
        self.W=self.comm.bcast(self.W,root=0)
        self.b=self.comm.bcast(self.b,root=0)
        # after receving self.W and self.b from rank 0, replace the original ones.
        for idx,rewrite_func in  enumerate(self.rewrite_Wb) :
            rewrite_func(self.W[idx],self.b[idx])      
        # clean up.
        self.W =  []
        self.b =  []

    def Save(self, savepath):
        # save trainable variables.
        if self.rank  == 0  :
            print("save  rank :",rank)
            for layer in  self.get_Wb :
                _W,_b = layer()
                self.W.append(_W)
                self.b.append(_b)
            trainable_vars = {"weights":self.W,"biases":self.b }
            print("save trainable veriables in  :", savepath)
            with open(savepath, 'wb') as pkfile:
                pickle.dump(trainable_vars, pkfile)
            print("done!")

            # clear up
            self.W=[]
            self.b=[]

 
    def Restore(self,savepath):
        # restore trainable variables
        print("restore trainable variables from :", savepath)
        with open(savepath, 'rb') as pkfile:
            trainable_vars  = pickle.load(pkfile)
        
        self.W = trainable_vars["weights"] 
        self.b = trainable_vars["biases"]       
        for idx,rewrite_func in  enumerate(self.rewrite_Wb) :
            rewrite_func(self.W[idx],self.b[idx])
        # clear up 
        self.W = []
        self.b = []
