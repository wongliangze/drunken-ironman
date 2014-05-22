from __future__ import division
import numpy as np
import Error

def init_param(initer,in_dims,location="unknown location"):
	"""
	    :Parameters:
	    initer: str
	    	Which initialization function to use
	    	examples: "zeros","random_uniform"
    	in_dims: int or tuple
    		Dimension(s) of array/vector to be initialized
    	location: str
    		Location of layer/sublayer calling this (optional; for error reporting)
	"""
	if initer == "zeros":
		return np.zeros(in_dims)		
	elif initer == "random_uniform":
		return random_uniform(in_dims)
	else:
		raise Error.InitError("Error.assign","Init function not defined/recognized in "+location)



def random_uniform(in_dims):
	"""
	Initialization of matrix
	in_dims = (rows,cols)
	"""

	rows,cols = in_dims
    r = np.sqrt(6)/np.sqrt(rows+cols+1)
    return random_state.rand(rows,cols)*2*r-r

