"""
Cost functions are scalar functions that take arrays as arguments.
"""
from __future__ import division
from Utils import KL_logistic,dKL_logistic,KL_tanh,dKL_tanh
import numpy as np

""" BASE FUNCTION CLASSES """

class param_function:
	"""
	Param functions take one or both params (w,b) as input.

	:Methods:
		deriv_w:
			derivative wrt w_mat
		deriv_b:
			derivative wrt b_vec
	"""	
	def __call__(self,*args):
		return 0.

	def deriv_w(self,*args):
		return 0.

	def deriv_b(self,*args):
		return 0.

class layer_function:
	"""
	Layer functions take at least a layer as input
	"""
	def __call__(self,*args):
		return 0.

	def delta(self,*args):
		return 0.

class hybrid_function(param_function,layer_function):
	pass


""" INHERITED FUNCTIONS """

class decay(param_function):
	def __call__(self,w,b=None):
		return 0.5*np.sum(w**2)

	def deriv_w(self,w,b=None):
		return 1.*w

class decay_weighted(param_function):
	def __call__(self,w,weights,b=None):
		return 0.5*np.sum(weights*(w**2))

	def deriv_w(self,w,weights,b=None):
		return 1.*w*weights 

class mean_squared_error(layer_function):
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
			return dKL_logistic(np.mean(layer,axis=0),sparse_rate)/sample_size
		else:
			sample_size = np.sum(1.-mask,axis=0)
			mask_size = np.shape(mask)[1]
			weight_mask = (1.-mask)/sample_size                 
			return np.dot(weight_mask, dKL_logistic(np.dot(weight_mask.T,layer), sparse_rate)/mask_size)

class sparsity_KL_tanh(layer_function):
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
			return np.sum(KL_tanh(np.dot(layer.T,1.-mask)/sample_size, sparse_rate))/mask_size

	def delta(self, layer, sparse_rate, mask = None):
		if mask == None:
                 sample_size = np.shape(layer)[0]
                 return dKL_tanh(np.mean(layer,axis=0),sparse_rate)/sample_size
		else:
			sample_size = np.sum(1.-mask,axis=0)
			mask_size = np.shape(mask)[1]
			return dKL_tanh(np.dot(layer.T,1.-mask)/sample_size, sparse_rate)/(mask_size*sample_size)

class contractive(hybrid_function):
	pass

class p_norm(layer_function):
    def __call__(self,layer,p):
        pass