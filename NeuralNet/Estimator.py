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
from scipy.optimize import fmin_l_bfgs_b as minimize
import time

import Error
import Namer
import Transfer
import Init
#from sklearn.base import BaseEstimator


class Layer(object):
    """
    Abstract Neural Layer class

    :Parameters:
        size_in: int
            Number of inputs (neurons in previous layer)
        size_out: int
            Number of outputs (neurons in current layer)        
        transfer: function
            transfer function from Transfer
        number: int
        	Position of current layer in a neural network (optional; for error reporting)
    	partition_in, partition_out: list
    		List of tuples indicating sublayers.

    """
    pass


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
        if np.shape(b_vec) != (self.size_out,): # b_vec should be a 1d array i.e. vector
            raise Error.InitError("Param dimension mismatch @ " + self.id)
    	self.w_mat = w_mat
    	self.b_vec = b_vec
    	
        # Sublayer transfer function
    	try:
            self.transfer = Transfer.assign(transfer)
        except NameError as error:
            raise Error.InitError(error.message + " in transfer func. assignment @ " + self.id)

    def pre_transfer(self, data_in):
        """ Affine transformation before applying transfer function step """        
        func_name = "pre_transfer"
        try:
            return np.dot(data_in,self.w_mat)+self.b_vec
        except Exception as error:
            raise Error.EvalError(error.message + " in execution of " + func_name + " @ " + self.id)                      

    def feed_forward(self, data_in):
        """ Feed forward step """        
        func_name = "feed_forward"
        try:
            return self.transfer(self.pre_transfer(data_in))    
        except Exception as error:
            raise Error.EvalError(error.message + " in execution of " + func_name + " @ " + self.id)                        

    def backprop(self, delta, data_in = None, data_out = None):
        """ Backpropagation of deltas """
        func_name = "backprop"        
        try:
            return np.dot(delta*self.transfer.deriv(self.pre_transfer(data_in),data_out), self.w_mat.T)
        except Exception as error:
            raise Error.EvalError(error.message + " in execution of " + func_name + " @ " + self.id)              

    def deriv_w(self, delta, data_in = None, data_out = None):
        """ Derivative w.r.t. w_mat """        
        func_name = "deriv_w"
        try:
            return np.dot(data_in.T,delta*self.transfer.deriv(self.pre_transfer(data_in),data_out))
        except Exception as error:
            raise Error.EvalError(error.message + " in execution of " + func_name + " @ " + self.id)

    def deriv_b(self, delta, data_in = None, data_out = None):
        """ Derivative w.r.t. b_vec """
        func_name = "deriv_b"
        try:
            return np.sum(delta*self.transfer.deriv(self.pre_transfer(data_in),data_out), axis = 0)
        except Exception as error:
            raise Error.EvalError(error.message + " in execution of " + func_name + " @ " + self.id)