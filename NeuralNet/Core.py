# -*- coding: utf-8 -*-
"""
Created on Mon Jun 09 15:42:40 2014

@author: Liang Ze
"""

from __future__ import division
import numpy as np
import itertools


import Cost
import Transfer
from Utils import init_random_uniform, getfree, setfree, None2Blank, listify

class Affine(object):
    """
    Affine transform.

    :Parameters:        
        w_mat: array of shape (M,N) 
            weight matrix 
        b_vec: array of shape (N,)
            bias vector; defaults to 0 if not specified
        w_free: array of shape (M,N), or 1 or 0            
            binary; 1 indicates free variables (as opposed to fixed); defaults to 1 if not specified
        b_free = array of shape (N,), or 1 or 0
            binary; 1 indicates free variables (as opposed to fixed); defaults to 1 if not specified and b_vec is specified;
            if both b_vec and b_free are unspecified, defaults to 0.         
    """	
    
    def __init__(self, w_mat, b_vec = None, w_free = 1, b_free = 1):
        self.w_mat = w_mat        
        self.w_free = w_free        

        if b_vec == None:
            self.b_vec = np.zeros(np.shape(w_mat)[1])
            self.b_free = 0
        else:
            self.b_vec = b_vec
            self.b_free = b_free
    
    @classmethod
    def init_by_size(cls, size_in, size_out, 
                     mtd = "rand_w", w_free = 1, b_free =1, random_seed = None):
        """
        :Parameters:
            mtd: string
                Initialization method. 'rand_w' or 'zeros'. Defaults to 'rand_w', which sets b = 0, w = random.
        """
        if mtd == "rand_w":        
            w_mat = init_random_uniform(size_in,size_out,random_seed)
        elif mtd == "zeros":
            w_mat = np.zeros((size_in,size_out))    
        else:
            raise NameError("Init. method not defined/recognized")
        b_vec = np.zeros(size_out)
        return cls(w_mat,b_vec,w_free,b_free)
        
    def feed_forward(self, data_in):
        """ Affine transformation before applying transfer function step """        
        #func_name = "pre_transfer"        
        return np.dot(data_in,self.w_mat)+self.b_vec        
    
    def backprop(self, delta):
        """ Backpropagation of deltas """
        #func_name = "backprop"                
        return np.dot(delta, self.w_mat.T)

    def deriv_w(self, delta, data_in):
        return np.dot(data_in.T,delta)

    def deriv_b(self, delta):        
        return np.sum(delta, axis = 0)

class AffineLayer(object):
    """
    Affine transfer layer.
    :Parameters:
        blocks_in: list of sublists
            Each sublist contains the nodes that belong to an input block
        blocks_out: list of sublists
            Each sublist contains the nodes that belong to an output block
        connections: list of pairs
            Each pair specifies an input block and output block
        affine_kwargs: list of dicts
            List of same size as "connections". Specifies the affine transform for each connection.            
    """ 

    def __init__(self, blocks_in, blocks_out, connections, affines):
        self.blocks_in = blocks_in
        self.blocks_out = blocks_out
        self.connections = connections        
        self.affines = affines
        
        self.block_sizes_in = [len(block) for block in blocks_in]
        self.block_sizes_out = [len(block) for block in blocks_out]

        self.size_in = max(max(blocks_in))+1
        self.size_out = max(max(blocks_out))+1
        
        self.affine_idx = [[None for j in blocks_out] for i in blocks_in]
        for idx, cxn in enumerate(connections):
            self.affine_idx[cxn[0]][cxn[1]] = idx
        
    @classmethod
    def init_by_size(cls, block_sizes_in, block_sizes_out, 
                     connections = None, complete = False,
                     mtds = None, w_frees = None, b_frees = None,
                     random_seed = None):
        """
        :Parameters:
            block_sizes_in,out: list of int
                Size of each disjoint block in the layer
            connections: list of pairs
                Each pair states the input and output blocks. If None, assigns 1-1, 1-Many, Many-1, or bipartite complete if complete=True.
            complete: bool
                Specifies whether to use a complete bipartite structure if connections are not specified.
            mtds: list of strings
                Init. method for each affine. 'rand_w' or 'zeros'. If None, assumes all 'rand_w'.
            w_frees: list
                w_free for each affine. If None, assumes all 1.
            b_frees: list
                b_free for each affine. If None, assumes all 1.
        """
        
        start_num_in = np.cumsum([0] + block_sizes_in[:-1])
        start_num_out = np.cumsum([0] + block_sizes_out[:-1])              
        blocks_in = [list(n+range(size)) for n,size in zip(start_num_in, block_sizes_in)]         
        blocks_out = [list(n+range(size)) for n,size in zip(start_num_out, block_sizes_out)]        
        
        if connections == None:
            if len(block_sizes_in) == len(block_sizes_out):
                # 1 to 1
                connections = [[i,i] for i in range(len(block_sizes_in))]
            elif len(block_sizes_in) == 1:
                # 1 to M
                connections = [[0,i] for i in range(len(block_sizes_out))]
            elif len(block_sizes_out) == 1:
                # M to 1
                connections = [[i,0] for i in range(len(block_sizes_in))]
            elif complete:
                # M to N, complete bipartite
                connections = [[i,j] for i,j in itertools.product(range(len(block_sizes_in)), range(len(block_sizes_out)))]
            else:
                raise TypeError("Ambiguous connection definition in init_by_size")
        
        if mtds == None:
            mtds = ["rand_w"]*len(connections)
        if w_frees == None:
            w_frees = [1.]*len(connections)
        if b_frees == None:
            b_frees = [1.]*len(connections)
        
        affines = [Affine.init_by_size(block_sizes_in[cxn[0]], block_sizes_out[cxn[1]], mtd, w_free, b_free, random_seed) \
                    for cxn, mtd, w_free, b_free in zip(connections, mtds, w_frees, b_frees)]
        
        return cls(blocks_in, blocks_out, connections, affines)       
    
    def get_affine_props(self, attr):
        """
        Get properties of affine layer.
        
        :Parameters:
            attr: string
                'w_mat', 'b_vec', 'w_free', 'b_free'
        """
        return [getattr(affine,attr) for affine in self.affines]
                
    def feed_forward(self, data_in):    
        data_out = np.zeros((len(data_in),self.size_out))
        for cxn,affine in zip(self.connections,self.affines):
            data_out[:,self.blocks_out[cxn[1]]] += affine.feed_forward(data_in[:,self.blocks_in[cxn[0]]])
        return data_out

    def backprop(self, delta):
        prev_delta = np.zeros((len(delta),self.size_in))
        for cxn,affine in zip(self.connections,self.affines):
            prev_delta[:,self.blocks_in[cxn[0]]] += affine.backprop(delta[:,self.blocks_out[cxn[1]]])            
        return prev_delta

    def deriv_w(self, delta, data_in):
        return [affine.deriv_w(delta[:,self.blocks_out[cxn[1]]] , data_in[:,self.blocks_in[cxn[0]]])
                for cxn,affine in zip(self.connections,self.affines)]

    def deriv_b(self, delta):
        return [affine.deriv_b(delta[:,self.blocks_out[cxn[1]]])
                for cxn,affine in zip(self.connections,self.affines)]

class TransferLayer(object):
    """
    Transfer function layer.
    :Parameters:
        blocks: list of sublists
            Each sublist contains the nodes that belong to a block
        transfers: list of strings
            Specifies transfer function for each block
    """ 
    def __init__(self, blocks, transfers):
        self.blocks = blocks
        self.transfers = [Transfer.assign(f) for f in transfers]
        
        self.size = max(max(blocks))+1
        self.block_sizes = [len(block) for block in blocks]
        
    @classmethod    
    def init_by_size(cls,block_sizes,transfers):
        """
        :Parameters:        
            block_sizes: list of int
                size of each disjoint block in the layer
            transfers: list of strings
                each string specifies the transfer function for each block
        """        
        start_num = np.cumsum([0] + block_sizes[:-1])
        blocks = [list(n+range(size)) for n,size in zip(start_num, block_sizes)]       
        if type(transfers) == str:
            transfers = [transfers]
        if len(transfers) == 1:
            transfers = transfers * len(blocks)        
        return cls(blocks,transfers)     
        
    def feed_forward(self,data_in):
        data_out = np.zeros_like(data_in)
        for block,f in zip(self.blocks,self.transfers):
            data_out[:,block] = f(data_in[:,block])
        return data_out

    def backprop(self, delta, data_in = None, data_out = None):
        """ Backpropagation of deltas """ 
        prev_delta = np.zeros_like(delta)        
        for block,f in zip(self.blocks,self.transfers):
            x = data_in[:,block] if (data_in != None) else None
            y = data_out[:,block] if (data_out != None) else None            
            prev_delta[:,block] = delta[:,block]*f.deriv(x,y)
        return prev_delta                  

class CostWrapper(object):    
    """
    Cost functions and their parameters.
    :Parameters:
        name: string
            Name of cost function
        weight: float
            Weightage of this cost component
        layer_id: int
            Id of parent layer
        in_id, out_id: int
            Id of input and output blocks of affine layer
        input_block, target_block: list of int
            Indexes of input and target blocks required
        xparams: dict
            Additional parameters for cost functiob (e.g. sparse rate)
    """
    def __init__(self, name, weight = 1., layer_id = None, in_id = None, out_id = None, input_block = None, target_block = None, xparams = None):        
        self.name = '{}.{}.{} {}'.format(*None2Blank([layer_id, in_id, out_id, name]))
        self.func = Cost.assign(name)
        self.type = self.func.type        
        
        self.weight = weight
        self.layer_id = layer_id
        self.in_id = in_id
        self.out_id = out_id
        self.input_block = input_block
        self.target_block = target_block
        self.xparams = xparams or {}    
    
    def compute(self, parent_layer, layer_data, **input_target_kwargs):
        """
        :Parameters:
            parent_layer: Layer object
                Layer on which this cost function is defined.
            layer_data: array
                Data from current layer
            **input_target_kwargs: keyed args from the following list
                Input_data, Input_mask, Target_data, Target_mask
        """
        if self.type == 'param':
            # Set up arguments
            affine_id = parent_layer.AffineLayer.affine_idx[self.in_id][self.out_id]
            if affine_id is None:
                raise NameError('Cannot apply cost ({}) to non-existing connection'.format(self.name))
            affine = parent_layer.AffineLayer.affines[affine_id]
            w = affine.w_mat
            b = affine.b_vec
            kwargs = self.xparams
            # Evaluate this component of the cost
            return self.weight * self.func(w,b, **kwargs)
        elif self.type == 'layer':            
            # Set up arguments            
            layer_block = parent_layer.TransferLayer.blocks[self.out_id]
            kwargs = {}
            keys = ['Input_data', 'Input_mask', 'Target_data', 'Target_mask']
            blocks = ['input_block', 'input_block', 'target_block', 'target_block']
            for key,block in zip(keys,blocks):
                if input_target_kwargs.has_key(key):
                    if input_target_kwargs[key] is not None:                        
                        if getattr(self,block) is not None:
                            kwargs[key] = input_target_kwargs[key][:,getattr(self,block)]
                        else:
                            kwargs[key] = input_target_kwargs[key]
            kwargs.update(self.xparams)
            # Evaluate this component of the cost
            return self.weight * self.func(layer_data[:,layer_block], **kwargs)
        else:
            raise NameError("Cost function not implemented")      
    
    def delta(self, parent_layer, layer_data = None, **input_target_kwargs):
        if 'delta' in dir(self.func):
            # Set up arguments
            layer_block = parent_layer.TransferLayer.blocks[self.out_id]
            kwargs = {}
            keys = ['Input_data', 'Input_mask', 'Target_data', 'Target_mask']
            blocks = ['input_block', 'input_block', 'target_block', 'target_block']
            for key,block in zip(keys,blocks):
                if input_target_kwargs.has_key(key):
                    if input_target_kwargs[key] is not None:                        
                        if getattr(self,block) is not None:
                            kwargs[key] = input_target_kwargs[key][:,getattr(self,block)]
                        else:
                            kwargs[key] = input_target_kwargs[key]
            kwargs.update(self.xparams)            
            # Evaluate this component of the gradient
            return self.weight * self.func.delta(layer_data[:,layer_block], **kwargs)
        else:
            return 0.
    
    def deriv_wb(self, parent_layer):
        if ('deriv_w' in dir(self.func)) and ('deriv_b' in dir(self.func)):
            # Set up arguments
            affine_id = parent_layer.AffineLayer.affine_idx[self.in_id][self.out_id]
            affine = parent_layer.AffineLayer.affines[affine_id]
            w = affine.w_mat
            b = affine.b_vec
            kwargs = self.xparams
            return self.weight * self.func.deriv_w(w,b,**kwargs), self.weight * self.func.deriv_b(w,b,**kwargs)
        else:
            return 0.,0.
            

class Layer(object):
    def __init__(self, AffineLayer, TransferLayer, Costs, layer_id = None):      
        """
        :Parameters:
            AffineLayer: object

            TransferLayer: object

            Costs: list of CostWrapper objects
        """
        self.AffineLayer = AffineLayer
        self.TransferLayer = TransferLayer
        self.Costs = Costs
        self.id = layer_id
        
        self.num_free = 0.
        self.num_free_list = []
        for affine in self.AffineLayer.affines:
            num_w = np.sum(np.ones_like(affine.w_mat)*affine.w_free)
            num_b = np.sum(np.ones_like(affine.b_vec)*affine.b_free)
            self.num_free += num_w + num_b
            self.num_free_list.append((num_w,num_b))

    @classmethod
    def init_by_size(cls, block_sizes_in, block_sizes_out, transfers, 
                     cost_params = None, connections = None, complete = False,
                     mtds = None, w_frees = None, b_frees = None,
                     random_seed = None, layer_id = None):
        affine = AffineLayer.init_by_size(block_sizes_in, block_sizes_out, connections, complete, mtds, w_frees, b_frees, random_seed)
        transfer = TransferLayer.init_by_size(block_sizes_out,transfers)               
        cost_list = []
        for param in (cost_params or []):    
            param['layer_id'] = layer_id
            if not param.has_key('target_block'):
                param['target_block'] = affine.blocks_out[param['out_id']]                
            cost_list.append(CostWrapper(**param))
        return cls(affine,transfer,cost_list,layer_id)

    def get_free_params(self, ):
        # only get free portion of parameters, in a flattened list            
        ravelled_params = []
        for affine in self.AffineLayer.affines:            
            ravelled_params += getfree(affine.w_mat,affine.w_free) + getfree(affine.b_vec,affine.b_free)                      
        return ravelled_params
    def set_free_params(self, new_params):
        # Inverse of get_free_params
        params = np.array(new_params)
        counter = 0
        for nums,affine in zip(self.num_free_list, self.AffineLayer.affines):  
            num_w = nums[0]
            num_b = nums[1]            
            setfree(affine.w_mat, affine.w_free, params[counter:counter+num_w])
            counter = counter + num_w
            setfree(affine.b_vec, affine.b_free, params[counter:counter+num_b])
            counter = counter + num_b            
            
    def get_affine_props(self, attr):
        """
        Get properties of affine layer.
        
        :Parameters:
            attr: string
                'w_mat', 'b_vec', 'w_free', 'b_free'
        """
        return [getattr(affine,attr) for affine in self.AffineLayer.affines]    
    def get_cost_props(self,attr):
        """
        Get properties of costs defined on this layer.
        
        :Parameters:
            attr: string
                e.g. 'name', 'type', 'weight', 'xparams'
        """
        return [getattr(cost,attr) for cost in self.Costs]
    def get_transfer_props(self,attr):
        """
        Get properties of transfer functions.
        
        :Parameters:
            attr: string
                e.g. 'block_sizes', 'blocks', 'transfers'                   
        """
        return [getattr(transfer,attr) for transfer in self.TransferLayer.transfers]
        
    def feed_forward(self,data_in, data_only = True, **input_target_kwargs):
        data_out = self.TransferLayer.feed_forward(
            self.AffineLayer.feed_forward(
                data_in))
        if data_only:
            return data_out
        
        cost_dict = {}
        cost_sum = 0.
        for cost in self.Costs:
            subcost = cost.compute(self, data_out, **input_target_kwargs)
            cost_dict[cost.name] = subcost
            cost_sum += subcost
        return data_out, cost_sum, cost_dict         

    def backprop(self,delta,data_in,data_out, ravelled=True, **input_target_kwargs):        
        pre_transfer_data = self.AffineLayer.feed_forward(data_in)                        
        # delta might be a scalar (e.g. 0) or a matrix of the same shape as data_out
        post_transfer_delta = np.ones(np.shape(data_out))*delta                 
        
        # Add deltas from this layer's cost functions        
        for cost in self.Costs:            
            post_transfer_delta[:,self.TransferLayer.blocks[cost.out_id]] += cost.delta(self, data_out, **input_target_kwargs)

        # Backprop through the Transfer layer
        pre_transfer_delta = self.TransferLayer.backprop(
            post_transfer_delta,
            pre_transfer_data,
            data_out)    

        # Pass delta into derivatives
        deriv_w = self.AffineLayer.deriv_w(pre_transfer_delta, data_in)
        deriv_b = self.AffineLayer.deriv_b(pre_transfer_delta)

        # Add derivs from this layer's cost functions
        for cost in self.Costs:
            if (cost.in_id is not None) and (cost.out_id is not None):
                dw, db = cost.deriv_wb(self)                
                # Add derivative to the respective affine params
                affine_id = self.AffineLayer.affine_idx[cost.in_id][cost.out_id]                            
                deriv_w[affine_id] += dw
                deriv_b[affine_id] += db

        # Backprop through Affine layer
        prev_delta = self.AffineLayer.backprop(pre_transfer_delta)
        
        if ravelled:
            ravelled_grads = []            
            for affine, dw, db in zip(self.AffineLayer.affines,deriv_w,deriv_b):
                ravelled_grads += getfree(dw,affine.w_free) + getfree(db,affine.b_free)             
            return prev_delta, ravelled_grads

        return prev_delta, deriv_w, deriv_b

class Net(object):
    def __init__(self, Layers):
        self.num_layers = len(Layers)
        self.Layers = Layers

        self.num_free_list = []
        for layer in self.Layers:       
            self.num_free_list.append(layer.num_free)

    @classmethod
    def init_by_size(cls, block_sizes, transfers, 
                     cost_params = None, connections = None, complete = False,
                     mtds = None, w_frees = None, b_frees = None,
                     random_seed = None):
        num_layers = len(block_sizes) - 1
        block_sizes_in_list = block_sizes[:-1]
        block_sizes_out_list = block_sizes[1:]
        cost_params_list = listify(cost_params,num_layers)
        transfers_list = listify(transfers, num_layers)
        connections_list = listify(connections, num_layers)
        complete_list = listify(complete, num_layers)
        mtds_list = listify(mtds, num_layers)
        w_frees_list = listify(w_frees, num_layers)
        b_frees_list = listify(b_frees, num_layers)
        layer_id_list = range(num_layers)
        random_seed_list = listify(random_seed,num_layers)

        keys_list = ['block_sizes_in','block_sizes_out','transfers', 
            'cost_params','connections', 'complete', 
            'mtds', 'w_frees','b_frees',
            'random_seed','layer_id']
        layers = []
        for vals_list in zip(block_sizes_in_list, block_sizes_out_list, transfers_list, 
            cost_params_list, connections_list, complete_list, 
            mtds_list, w_frees_list, b_frees_list,
            random_seed_list, layer_id_list):
            
            layer_kwargs = {key:val for key,val in zip(keys_list,vals_list)}
            layers.append(Layer.init_by_size(**layer_kwargs))

        return cls(layers)

    def get_free_params(self, ):
        # only get free portion of parameters, in a flattened list            
        ravelled_params = []
        for layer in self.Layers:            
            ravelled_params += layer.get_free_params()
        return np.array(ravelled_params)
    def set_free_params(self, new_params):
        # Inverse of get_free_params
        params = np.array(new_params)
        counter = 0
        for num, layer in zip(self.num_free_list, self.Layers):
            layer.set_free_params(params[counter:counter+num])
            counter = counter + num         

    def costgrad(self, Input_data, Target_data, Input_mask = None, Target_mask = None, cost_only = False):
        input_target_kwargs = {'Input_data': Input_data, 'Input_mask': Input_mask, 'Target_data': Target_data, 'Target_mask':Target_mask}
        
        # Feed forward to compute cost
        data_list = [None]*(self.num_layers + 1)
        total_cost= 0        

        data_list[0] = Input_data        
        for i in range(self.num_layers):
            data_out, cost_sum, cost_dict = self.Layers[i].feed_forward(data_list[i], data_only = False, **input_target_kwargs)
            data_list[i+1] = data_out                        
            total_cost += cost_sum

        if cost_only:
            return total_cost

        # Backpropagate to compute grads        
        delta = 0.
        ravelled_grads = []
        for i in range(self.num_layers - 1, -1 , -1):            
            delta, layer_grads = self.Layers[i].backprop(delta, data_list[i], data_list[i+1], **input_target_kwargs)            
            ravelled_grads = layer_grads + ravelled_grads

        return total_cost, ravelled_grads      

    def feed_forward(self, Input_data, Target_data = None, Input_mask = None, Target_mask = None):
        input_target_kwargs = {'Input_data': Input_data, 'Input_mask': Input_mask, 'Target_data': Target_data, 'Target_mask':Target_mask}
        
        data_list = [None]*(self.num_layers + 1)
        data_list[0] = Input_data        
        for i in range(self.num_layers):
            data_list[i+1] = self.Layers[i].feed_forward(data_list[i], data_only = True, **input_target_kwargs)
        return data_list

    def predict(self, Input_data, Target_data = None, Input_mask = None, Target_mask = None):
        input_target_kwargs = {'Input_data': Input_data, 'Input_mask': Input_mask, 'Target_data': Target_data, 'Target_mask':Target_mask}
        data = Input_data
        for layer in self.Layers:
            data = layer.feed_forward(data,data_only=True,**input_target_kwargs)
        return data
    
    def score(self, Input_data, Target_data, score_func = 'MSE', Input_mask = None, Target_mask = None):
        input_target_kwargs = {'Input_data': Input_data, 'Input_mask': Input_mask, 'Target_data': Target_data, 'Target_mask':Target_mask}
        prediction = self.predict(**input_target_kwargs)
        f = Cost.assign(score_func)
        return f(prediction, **input_target_kwargs)

    def cost_breakdown(self, Input_data, Target_data, Input_mask = None, Target_mask = None):
        input_target_kwargs = {'Input_data': Input_data, 'Input_mask': Input_mask, 'Target_data': Target_data, 'Target_mask':Target_mask}
        
        # Feed forward to compute cost        
        final_cost_dict= {}
        data = Input_data

        for layer in self.Layers:
            data, cost_sum, cost_dict = layer.feed_forward(data, data_only = False, **input_target_kwargs)
            final_cost_dict.update(cost_dict)            

        return final_cost_dict

