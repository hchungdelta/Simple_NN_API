import numpy as np

class activation():
    def ReLU(self,inp_layer):
        """
        input size :  batch ,filters, y ,x
        """
        output=np.zeros_like(inp_layer)
        output = np.maximum.reduce([inp_layer,output])
        self.output = output
        return output
    def derv_ReLU(self):
        """
        input size : batch, filters, y ,x
        """
        output = np.where(self.output!=0,1,0)
        return output

class CNN_layer(activation):
    def __init__(self,inp_layer_size,stride=(1,1),kernel_size=(2,2),paddling =True ,activation = "None"):
        """
        arguments
        - inp_layer_size : shape= (output_filters ,input filters, y ,x) shape, if MNIST it is   output_filters * 1 *  28 * 28
        - kernel_size    : size of kernels(filters), default is (2,2)
        - stride         : displacement of one step, default is (1,1)
        - paddling       : if stride_x or _y = 1 : add zero array at boundary, in order to maintain the size of inp_layer
                           if stirde_x or _y > 1 : add zero array at bountary, in order not to miss boundary.
        - activation     : ReLU or None.
        """
        self.amount_of_output_filters,self.amount_of_input_filters= inp_layer_size[:2]
        self.inpy_size, self.inpx_size = inp_layer_size[2:]
        self.kernel_y , self.kernel_x  = kernel_size
        self.stride_y , self.stride_x  = stride
        self.paddling   = paddling
        self.activation = activation
        # select activation function
        if activation == "ReLU":
            self.act_func=self.ReLU
            self.dact_func=self.derv_ReLU
        # create initial random kernels and bais
        if self.paddling :
            if self.stride_y == 1 : 
                outy_size=self.inpy_size
            if self.stride_y > 1 :
                outy_size=1+int(np.ceil((self.inpy_size-self.kernel_y)/self.stride_y) )
            if self.stride_x == 1 :
                outx_size=self.inpx_size
            if self.stride_x > 1 :
                outx_size=1+int(np.ceil((self.inpx_size-self.kernel_x)/self.stride_x ) ) # np.ceil(1.3)=2
        else :
            outy_size=1+int((self.inpy_size-self.kernel_y)/self.stride_y  )    # int(1.3)=1
            outx_size=1+int((self.inpx_size-self.kernel_x)/self.stride_x  )

        self.kernel =  np.random.normal(0.5,scale=0.50,size=(self.amount_of_output_filters,self.amount_of_input_filters,self.kernel_y,self.kernel_x))

        init_bias = np.random.random(self.amount_of_output_filters)
        self.bias= np.repeat(init_bias,outy_size*outx_size ).reshape(self.amount_of_output_filters,outy_size,outx_size)
        #self.bias   = np.full((output_filter_amount,outy_size,outx_size),fill_value=np.random.random())

    def forward(self,inp_layer) :
        #self.SUM_inp = np.sum(inp_layer)  # sum of all values in inp ... 
        self.SUM_inp = np.sum(abs(inp_layer))
        self.inp_layer      = inp_layer   # non paddling input  
        self.batch  = inp_layer.shape[0]  # batch size
        self.de_paddling = (0,0)          # for backprop usage

        _,_,inpy_size, inpx_size = self.inp_layer.shape  

        if self.paddling :
            if self.stride_y == 1:
                add_paddling_y  = self.stride_y* ( self.kernel_y - 1)   
            if self.stride_y > 1 :
                add_paddling_y = int(np.ceil((self.inpy_size-self.kernel_y)/self.stride_y))
            if self.stride_x == 1 :
                add_paddling_x  = self.stride_x *( self.kernel_x - 1)   
            if self.stride_x > 1 :
                add_paddling_x = int(np.ceil((self.inpx_size-self.kernel_x)/self.stride_x))

            both_boundary_y = add_paddling_y // 2    # add paddling on both boundary
            front_only_y    = add_paddling_y % 2     # add one more paddling on front boundary if the amount needed to add if odds
            both_boundary_x = add_paddling_x // 2
            front_only_x    = add_paddling_x % 2
            # add zero paddling along y-axis
            self.inp_layer=np.concatenate(
                   (np.zeros((self.batch,self.amount_of_input_filters,front_only_y+both_boundary_y,inpx_size)),
                    self.inp_layer ) ,axis=2)
            self.inp_layer=np.concatenate((self.inp_layer, 
                    np.zeros((self.batch,self.amount_of_input_filters, both_boundary_y ,inpx_size)))  ,axis=2)
            inpy_size = self.inp_layer.shape[2]   # update during paddling
            #  add zero paddling along x-axis
            self.inp_layer=np.concatenate(
                   (np.zeros((self.batch,self.amount_of_input_filters, inpy_size, both_boundary_x+front_only_x)),
                    self.inp_layer) ,axis=3)
            self.inp_layer=np.concatenate((self.inp_layer, 
                    np.zeros((self.batch,self.amount_of_input_filters, inpy_size, both_boundary_x)))  ,axis=3)
            inpx_size = self.inp_layer.shape[3]   # update during paddling

            # record how many paddlings were added.
            self.de_paddling = (add_paddling_y,add_paddling_x)            

        # paddling layer
        self.paddling_layer =  self.inp_layer

        self.total_step_y = 1+ int( (inpy_size -  self.kernel_y)/self.stride_y  )
        self.total_step_x = 1+ int( (inpx_size -  self.kernel_x)/self.stride_x  )

        # inputs multiply by weights
        output_all_filter=np.zeros(shape=[self.batch,self.amount_of_output_filters,self.total_step_y,self.total_step_x])
        for output_filter in range(self.amount_of_output_filters) :
            for input_filter in range(self.amount_of_input_filters) :
                for step_y in range(self.total_step_y):
                    for step_x in range(self.total_step_x) :
                        this_grid=self.inp_layer[:,input_filter,
                                                 self.stride_y*step_y:self.kernel_y+ self.stride_y*step_y,
                                                 self.stride_x*step_x:self.kernel_x+ self.stride_x*step_x]

                        weighted_grid = np.multiply(this_grid, np.tile(self.kernel[output_filter][input_filter],  (self.batch,1,1))  )
                        output_all_filter[:,output_filter,step_y,step_x]+= np.sum(weighted_grid.reshape(-1,self.kernel_x*self.kernel_y) ,axis=1 )

        # add bias         
        output_all_filter=np.add( np.array(output_all_filter) ,np.tile(self.bias,(self.batch,1,1,1) )   )

        #activation function
        if self.activation == "ReLU" :
            output_all_filter=self.act_func(output_all_filter)
        self.SUM_out = np.sum(abs(output_all_filter) )
        return output_all_filter
    def give_me_paddling_layer(self) :
        return self.paddling_layer

    def backprop(self,dL):
        # sum of input (x1,x2,x3 ... )
        if self.activation == "ReLU":
            dL =dL*self.dact_func()
        self.sum_dW=np.zeros_like(self.kernel)
        self.sum_db=np.zeros_like(self.bias)
        self.sum_dL=np.zeros_like(self.paddling_layer)
        # sum dW
        for output_filter in range(self.amount_of_output_filters):
            for input_filter in range(self.amount_of_input_filters ): 
                for y in range( self.kernel_y ) :
                     for x in range(self.kernel_x) :
                         weight_related_input  =self.paddling_layer[:,input_filter,y: y+self.paddling_layer.shape[2]+1-self.kernel_y:self.stride_y,
                                                                                   x: x+self.paddling_layer.shape[3]+1-self.kernel_x:self.stride_x]
                         self.sum_dW[output_filter][input_filter][y][x] = np.sum(np.multiply(dL[:,output_filter,:,:],weight_related_input)  )
        # sum db
        for output_filter in range(self.amount_of_output_filters) :
            self.sum_db[output_filter] = np.sum( np.full(dL[:,0,:,:].shape,fill_value=(np.sum(dL[:,output_filter,:,:])) )   ,axis=0)
        # sym dL
        for input_filter in range(self.amount_of_input_filters ):
            for y in range(self.total_step_y):
                for x in range(self.total_step_x):
                    single_dL        = np.repeat(dL[:,:,y,x],self.kernel_y*self.kernel_x).reshape(self.batch,self.amount_of_output_filters,self.kernel_y,self.kernel_x)
                    related_weight   = np.tile(self.kernel[:,input_filter],(self.batch,1,1,1))
                    self.sum_dL[:,input_filter, self.stride_y*y: self.stride_y*y +self.kernel_y,
                                                self.stride_x*x: self.stride_x*x +self.kernel_x] += np.sum( single_dL*related_weight , axis=1)
 
        cut_y , cut_x = self.de_paddling  
        self.sum_dL = self.sum_dL[:,:, int(np.ceil(cut_y/2)):self.sum_dL.shape[2]-int(cut_y/2),  int(np.ceil(cut_x/2)):self.sum_dL.shape[3]-int(cut_x/2)]
        return self.sum_dL               

    def  update(self,dW,db,lr) :  
        # prevent exponential gradent (due to the backprop algorthim of CNN)
        # implicitly  involve the "batchsize"
        normalized_factor_w = 1+abs(self.SUM_inp)*self.kernel_y*self.kernel_x*self.amount_of_input_filters   #*self.amount_of_output_filters
        normalized_factor_b=  1+abs(self.SUM_out)*self.amount_of_output_filters
        limitb = np.max(abs(db))/normalized_factor_b
        limitw = np.max(abs(dW))/normalized_factor_w
        limit = np.max([limitb,limitw])
        while limit< 0.01  :
                limit =limit *2
                db = db*2
                dW = dW*2
        while limit > (0.1/lr) :
            db = db/2
            dW = dW/2
            limit = limit/2 

        self.bias= self.bias - lr*(1/normalized_factor_b)*db
        self.kernel = self.kernel -lr*(1/normalized_factor_w)*dW
    def rewrite_Wb(self,kernel,bias):
        self.kernel = kernel
        self.bias = bias
    def get_Wb(self):
        return self.kernel , self.bias
    def get_dWb(self):
        return self.sum_dW, self.sum_db


 

class max_pooling:
    def __init__(self, kernel=(2,2),stride=(2,2)):
        self.stride_y,self.stride_x = stride
        self.size_y, self.size_x    = kernel

        
    def forward(self,inp_layer  ):
        """
        arguments
        - inp_layer :  in (batch_size,filters,y,x) shape
        """
        self.batch, self.amount_of_input_filters ,self.inpy_size, self.inpx_size = inp_layer.shape 
        self.total_step_y = 1+ int( (self.inpy_size -  self.size_y)/self.stride_y  )
        self.total_step_x = 1+ int( (self.inpx_size -  self.size_x)/self.stride_x  )

        # forward use:
        output_filter=np.zeros(shape=[self.batch,self.amount_of_input_filters,self.total_step_y,self.total_step_x])

        # for backprop use (to tell which output in the kernel contributes to the maxpool ( 1 or 0 )):
        mask_output_filter=np.zeros_like(inp_layer)

        for b in range(self.batch) :
            for this_filter in range(self.amount_of_input_filters) :
                for step_y in range(self.total_step_y):
                    for step_x in range(self.total_step_x):
                        y_start, y_end = [self.stride_y*step_y,self.stride_y*step_y+self.size_y]
                        x_start, x_end = [self.stride_x*step_x,self.stride_x*step_x+self.size_x]

                        this_grid=inp_layer[b,this_filter,y_start:y_end,x_start:x_end]
                        output_filter[b,this_filter,step_y,step_x]=np.max(this_grid) 
                        max_idy , max_idx = np.unravel_index( np.argmax(this_grid), np.array(this_grid).shape)
                        mask_output_filter[b,this_filter,y_start+max_idy,x_start+max_idx]=1

        self.mask = mask_output_filter
        return output_filter

    def backprop(self,dL):
        output_filter=np.zeros_like(self.mask)
        
        for step_y in range(self.total_step_y):
            for step_x in range(self.total_step_x):
                y_start, y_end = [self.stride_y*step_y,self.stride_y*step_y+self.size_y]
                x_start, x_end = [self.stride_x*step_x,self.stride_x*step_x+self.size_x] 
                this_grid=self.mask[:,:,y_start:y_end,x_start:x_end]
                dL_grid= np.repeat( dL[:,:,step_y,step_x] ,(self.size_y*self.size_x))
                dL_grid=np.reshape(dL_grid, (self.batch,self.amount_of_input_filters,self.size_y,self.size_x) )
                output_filter[:,:,y_start:y_end,x_start:x_end]+=this_grid*dL_grid #[:,:,step_y,step_x]
        return output_filter
                    
class flatten():
    def forward(self,inp_layer):
       self.inp_layer_shape = inp_layer.shape 
       return inp_layer.reshape(self.inp_layer_shape[0],-1)

    def backprop(self,dL) :
       dL=dL.reshape(self.inp_layer_shape)
       return dL
