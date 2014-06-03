"""
Created on Thu 22 May 2014 at 09:25

@author: WLZ

A modular neural network library with the following flexibilities:
	- Number of layers
	- Connection structures
	- Transfer functions (linear, logistic, tanh)
	- Layer functions (e.g. MSE, sparsity penalty)
	- Param functions (e.g. decay weight)

And the following functionalities:
	- Greedy layer-wise training
	- Fine-tuning
	- Hyperparameter search
	- Normalization
	- Display
	- Saving and loading

This is based on work by:
	- SW, DC: Original SAE
	- OCC: Inheritance from sklearn; object oriented framework
	- WY: stacked SAE; new connection structures; greedy layer-wise training
	- Neurolab: code structure and organization

All optimization will be carried out by scipy.optimize.fmin_l_bfgs_b .

If desired, we can use BaseEstimator from sklearn.base as the parent for Net. However, "object" is good enough for our purposes.
"""

from __future__ import division
import numpy as np
import time

import Error
import Namer
import Transfer
import Init

from Utils import merge_blocks

#from scipy.optimize import fmin_l_bfgs_b as minimize
#from sklearn.base import BaseEstimator

class TransferLayer(object):
    """
    Transfer function layer.

    :Parameters:        
        block_sizes: list of int
            size of each disjoint block in the layer
        transfers: list of strings
            each string specifies the transfer function for each block
        blocks: list of sublists (optional)
            each sublist specifies which nodes belong in a block of the partition
    """ 
    def __init__(self,block_size,transfers,blocks=None):
        self.size = np.sum(block_sizes)
        self.num_blocks = len(block_sizes)
        self.transfers = [Transfer.assign(f) for f in transfers]
        if blocks == None:
            start_num = np.cumsum([0] + block_sizes[:-1])
            self.blocks = [n+range(size) for n,size in zip(start_num, block_sizes)]
        else:
            self.blocks = blocks

    def feed_forward(self,data_in):
        data_out = np.zeros_like(data_in)
        for block,f in zip(self.blocks,self.transfers):
            data_out[:,block] = f(data_in[:,block])
        return data_out

    def backprop(self, delta, data_in = None, data_out = None):
        """ Backpropagation of deltas """ 
        prev_delta = np.zeros_like(delta)        
        for block,f in zip(self.blocks,self.transfers):
            x = data_in[:,block] if data_in else None
            y = data_out[:,block] if data_out else None            
            prev_delta[:,block] = delta[:,block]*f.deriv(x,y)
        return prev_delta

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
            self.b_vec = 0
            self.b_free = 0
        else:
            self.b_vec = b_vec
            self.b_free = b_free
    	
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
        affine_kwargs: list of dicts
            inputs for each sub-affine

        merge_by_block: list of sublists
            each sublists lists blocks to be merged    
    """ 
    def __init__(self, affine_kwargs, merge_by_block = None, merge_by_node = None):
        self.affines = [Affine(**kwargs) for kwargs in affine_kwargs]        
        self.connections = [[i,i] for i in range(len(self.affines))]

        block_sizes_in = np.array([np.shape(affine.w_mat)[0] for affine in self.affines])
        block_sizes_out = np.array([np.shape(affine.w_mat)[1] for affine in self.affines])

        # Merging
        if merge_by_block != None:
            for merge_set in merge_by_block['in']:
                assert len(set(block_sizes_in[merge_set])) == 1, "Merged blocks must have same size"
            for merge_set in merge_by_block['out']:
                assert len(set(block_sizes_out[merge_set])) == 1, "Merged blocks must have same size"

            merge_in = merge_blocks(merge_by_block['in'], len(self.affines)) or range(len(self.affines))
            merge_out = merge_blocks(merge_by_block['out'], len(self.affines)) or range(len(self.affines))
            
            self.connections = [ [merge_in[io[0]], merge_out[io[1]]] for io in self.connections]
            block_sizes_in = [block_sizes_in[merge_in.index(i)] for i in range(max(merge_in) + 1)]            
            block_sizes_out = [block_sizes_out[merge_out.index(i)] for i in range(max(merge_out) +1)]

        elif merge_by_node != None:
            assert False, "merge_by_node not implemented yet"
        
        
        start_num_in = np.cumsum([0] + block_sizes_in[:-1])
        start_num_out = np.cumsum([0] + block_sizes_out[:-1])

        self.size_in = np.sum(block_sizes_in)
        self.size_out = np.sum(block_sizes_out)            
        
        self.blocks_in = [n+range(size) for n,size in zip(start_num_in, block_sizes_in)] 
        self.blocks_out = [n+range(size) for n,size in zip(start_num_out, block_sizes_out)] 
        
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


class Sublayer(object):
    """
    Abstract Neural Layer class. Basic building block of Layer.

    :Parameters:
        transfer: str
            transfer function from Transfer
            examples: "linear","tanh","logistic"
        w_mat: array
            weight matrix of shape (M,N) 
        b_vec: array
            bias vector of shape (N,)         
        depth: int
            depth of the layer it belongs to 
            (optional; for error reporting)
        position: int
            position of this sublayer within its parent layer 
            (optional; for error reporting)

    """ 
    
    def __init__(self, transfer, w_mat, b_vec, depth = None, position = None):      
        # Sublayer identification attributes
        self.id = Namer.sublayer_name(depth,position)       
        self.depth = depth
        self.position = position

        # Sublayer parameter attributes
        self.size_in, self.size_out = np.shape(w_mat)                
        self.w_mat = w_mat
        self.b_vec = b_vec
        
        # Sublayer transfer function        
        self.transfer = Transfer.assign(transfer)        

    def pre_transfer(self, data_in):
        """ Affine transformation before applying transfer function step """        
        #func_name = "pre_transfer"        
        return np.dot(data_in,self.w_mat)+self.b_vec        

    def feed_forward(self, data_in):
        """ Feed forward step """        
        #func_name = "feed_forward"        
        return self.transfer(self.pre_transfer(data_in))            

    def backprop(self, delta, data_in = None, data_out = None):
        """ Backpropagation of deltas """
        #func_name = "backprop"                
        return np.dot(delta*self.transfer.deriv(self.pre_transfer(data_in),data_out), self.w_mat.T)        

    def deriv_w(self, delta, data_in = None, data_out = None):
        """ Derivative w.r.t. w_mat """        
        #func_name = "deriv_w"
        return np.dot(data_in.T,delta*self.transfer.deriv(self.pre_transfer(data_in),data_out))
        
    def deriv_b(self, delta, data_in = None, data_out = None):
        """ Derivative w.r.t. b_vec """
        #func_name = "deriv_b"
        return np.sum(delta*self.transfer.deriv(self.pre_transfer(data_in),data_out), axis = 0)        