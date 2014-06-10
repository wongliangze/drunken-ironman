# -*- coding: utf-8 -*-
"""
Created on Mon Jun 09 15:42:40 2014

@author: Liang Ze
"""

from __future__ import division
import numpy as np
import itertools

from Utils import init_random_uniform
import Cost

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
        self.transfers = transfers
        
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

def CostWrapper(object):    
    """
    Cost functions and their parameters.
    :Parameters:
        func_name: string
            Name of cost function
        weight: float
            Weightage of this cost component
        layer_input: int
            Index of block_out that it takes as input argument
        affine_input: pair of int
            Indexes of (block_in, block_out)
        xparams: dict
            Additional parameters for cost functiob (e.g. sparse rate)
    """
    def __init__(self, func_name, weight = 1., layer_id = None, in_id = None, out_id = None, xparams = None):
        self.name = '{}.{}.{} {}'.format(layer_id or '', in_id or '', out_id or '', func_name)
        self.func = Cost.assign(func_name)
        self.type = self.func.type        
        
        self.weight = weight
        self.layer_id = layer_id
        self.in_id = in_id
        self.out_id = out_id
        self.xparams = xparams or {}
    
    def __call__(self, parent_layer, layer_data = None, xkwargs = None):
        """
        :Parameters:
            parent_layer: Layer object
                Layer on which this cost function is defined.
            layer_data: array
                Data for layer functions.
            xkwargs: dict
                Extra kwargs.
        """
        if self.type == 'param':
            # Set up arguments
            affine_id = parent_layer.AffineLayer.affine_idx[self.in_id,self.out_id]
            affine = parent_layer.AffineLayer.affines[affine_id]
            w = affine.w_mat
            b = affine.b_vec
            kwargs = xkwargs or {}
            kwargs.update(self.xparams)
            # Evaluate this component of the cost
            return self.weight * self.func(w,b, **kwargs)
        elif self.type == 'layer':
            # Set up arguments
            block = parent_layer.TransferLayer.blocks[self.out_id]                      
            layer = layer_data[:,block]
            kwargs = xkwargs or {}
            kwargs.update(self.xparams)
            # Evaluate this component of the cost
            return self.weight * self.func(layer,**kwargs)
        else:
            raise NameError("Cost function not implemented")      
    
    def delta(self, parent_layer, layer_data = None, xkwargs = None):
        if 'delta' in dir(self.func):
            # Set up arguments
            block = parent_layer.TransferLayer.blocks[self.out_id]                      
            layer = layer_data[:,block]
            kwargs = xkwargs or {}
            kwargs.update(self.xparams)
            # Evaluate this component of the gradient
            return self.weight * self.func.delta(layer, **kwargs)
        else:
            return 0.
    
    def deriv_wb(self, parent_layer, layer_data = None, xkwargs = None):
        if 'deriv_w' in dir(self.func):
            # Set up arguments
            affine_id = parent_layer.AffineLayer.affine_idx[self.in_id,self.out_id]
            affine = parent_layer.AffineLayer.affines[affine_id]
            w = affine.w_mat
            b = affine.b_vec
            kwargs = xkwargs or {}
            kwargs.update(self.xparams)
            return self.weight * self.func.deriv_w(w,b,**kwargs), self.weight * self.func.deriv_b(w,b,**kwargs)
        else:
            return 0.
    
            
