"""
Cost functions are scalar functions that take arrays as arguments.
"""
from __future__ import division
from Utils import KL_logistic,dKL_logistic,KL_tanh,dKL_tanh
import numpy as np

param_func_list = ['DK','DK_weighted']
layer_func_list = ['MSE','KL_tanh','KL_logistic']
hybrid_func_list = []

cost_func_list = param_func_list + layer_func_list + hybrid_func_list

def assign(cost):
    	"""
	    :Parameters:
	    cost: str
	    	Which cost function to use
	    	examples: 'DK', 'DK_weighted', 'MSE', 'KL_tanh', 'KL_logistic'
	"""	
	if cost == "DK":
		return decay()
	elif cost == "DK_weighted":
		return decay_weighted()
	elif cost == "MSE":
	    return mean_squared_error()
	elif cost == "KL_logistic":
		return sparsity_KL_logistic()
	elif cost == "KL_tanh":
		return sparsity_KL_tanh()   
	else:
	    raise NameError("Cost function not defined/recognized")


""" BASE FUNCTION CLASSES """

class param_function(object):
	"""
	Param functions take one or both params (w,b) as input.

	:Methods:
		deriv_w:
			derivative wrt w_mat
		deriv_b:
			derivative wrt b_vec
	"""
	def __init__(self):	
		self.type = "param"

	def __call__(self,*args):
		return 0.

	def deriv_w(self,*args):
		return 0.

	def deriv_b(self,*args):
		return 0.

class layer_function(object):
	"""
	Layer functions take at least a layer as input
	"""
	def __init__(self):
		self.type = "layer"

	def __call__(self,*args):
		return 0.

	def delta(self,*args):
		return 0.

class hybrid_function(param_function,layer_function):
	# NOT SUPPORTED YET
	def __init__(self):
		super(hybrid_function,self).__init__()
		self.type = "hybrid"


""" INHERITED FUNCTIONS """

class decay(param_function):
	def __init__(self):
		super(decay,self).__init__()
		self.name = "DK"
	def __call__(self,w,b=None):
		return 0.5*np.sum(w**2)

	def deriv_w(self,w,b=None):
		return 1.*w

class decay_weighted(param_function):
	def __init__(self):
		super(decay_weighted,self).__init__()
		self.name = "DK_weighted"	
	def __call__(self,w,weights=1.,b=None):
		return 0.5*np.sum(weights*(w**2))

	def deriv_w(self,w,weights=1.,b=None):
		return 1.*w*weights 

class mean_squared_error(layer_function):
	def __init__(self):
		super(mean_squared_error,self).__init__()
		self.name = "MSE"	
	def __call__(self, layer, other=0, mask = None):
		"""
		:Parameters:
			layer: ndarray
				original layer, to which this function belongs
			other: ndarray; broadcastable with layer
				data for comparison
			mask: ndarray
				binary values
				0 or False  : not missing
				1 or True	: missing				
		"""
		if mask == None:
			sample_size = np.shape(layer)[0]
			mask = 0.
		else:
			sample_size = np.sum(1.-mask,axis=0)

		return np.sum(np.sum(0.5 * (1.-mask)*(layer - other)**2,axis=0)/sample_size)		

	def delta(self, layer, other=0, mask=None):
		if mask == None:
			sample_size = np.shape(layer)[0]
			mask = 0.
		else:
			sample_size = np.sum(1.-mask,axis=0)

		return (1.-mask)*(layer - other)/sample_size

class sparsity_KL_logistic(layer_function):
	def __init__(self):
		super(sparsity_KL_logistic,self).__init__()
		self.name = "KL_logistic"	
	def __call__(self, layer, sparse_rate, mask = None):
		"""
		:Parameters:
			layer: ndarray
				original layer, to which this function belongs
			sparse_rate: float
				targetted sparse rate
			mask: ndarray
				binary values
				0 or False  : not missing
				1 or True	: missing	

		Unmasked versions are equivalent to the masked versions with the following assignments:
			sample_size = np.shape(layer)[0]			
			mask_size = 1.
			mask = np.ones((sample_size,1))
		"""
		if mask == None:			
			return np.sum(KL_logistic(np.mean(layer,axis=0),sparse_rate))
		else:
			sample_size = np.sum(1.-mask,axis=0)
			mask_size = np.shape(mask)[1]
			weight_mask = (1.-mask)/sample_size
			return np.sum(KL_logistic(np.dot(weight_mask.T,layer), sparse_rate))/mask_size

	def delta(self, layer, sparse_rate, mask = None):
		if mask == None:
			sample_size = np.shape(layer)[0]
			return np.outer(np.ones(sample_size),dKL_logistic(np.mean(layer,axis=0),sparse_rate)/sample_size)
		else:
			sample_size = np.sum(1.-mask,axis=0)
			mask_size = np.shape(mask)[1]
			weight_mask = (1.-mask)/sample_size                 
			return np.dot(weight_mask, dKL_logistic(np.dot(weight_mask.T,layer), sparse_rate)/mask_size)

class sparsity_KL_tanh(layer_function):
	def __init__(self):
		super(sparsity_KL_tanh,self).__init__()
		self.name = "KL_tanh"	
	def __call__(self, layer, sparse_rate, mask = None):
		"""
		:Parameters:
			layer: ndarray
				original layer, to which this function belongs
			sparse_rate: float
				targetted sparse rate
			mask: ndarray
				binary values
				0 or False  : not missing
				1 or True	: missing	

		Unmasked versions are equivalent to the masked versions with the following assignments:
			sample_size = np.shape(layer)[0]			
			mask_size = 1.
			mask = np.ones((sample_size,1))
		"""
		if mask == None:			
			return np.sum(KL_tanh(np.mean(layer,axis=0),sparse_rate))
		else:
			sample_size = np.sum(1.-mask,axis=0)
			mask_size = np.shape(mask)[1]
			weight_mask = (1.-mask)/sample_size
			return np.sum(KL_tanh(np.dot(weight_mask.T,layer), sparse_rate))/mask_size

	def delta(self, layer, sparse_rate, mask = None):
		if mask == None:			
			sample_size = np.shape(layer)[0]
			return np.outer(np.ones(sample_size),dKL_tanh(np.mean(layer,axis=0),sparse_rate)/sample_size)
		else:
			sample_size = np.sum(1.-mask,axis=0)
			mask_size = np.shape(mask)[1]
   			weight_mask = (1.-mask)/sample_size  
			return np.dot(weight_mask, dKL_tanh(np.dot(weight_mask.T,layer), sparse_rate)/mask_size)

class contractive(hybrid_function):
	pass

class p_norm(layer_function):
    def __call__(self,layer,p):
        pass