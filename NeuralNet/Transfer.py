import Error

# Supported list of transfer functions
transfer_list = ["linear","tanh","logistic"]

def assign(transfer):
	"""
	    :Parameters:
	    transfer: str
	    	Which transfer function to use
	    	examples: "linear","tanh","logistic"
    	location: str
    		Location of layer/sublayer calling this (optional; for error reporting)
	"""	
	if transfer == "linear":
		return identity
	elif transfer == "tanh":
		return tanh_sigmoid
	elif transfer == "logistic":
		return logistic_sigmoid
	else:
		raise NameError("Transfer function not defined/recognized")

class identity:
    """
    Identity transfer function

    :Parameters:
        x: ndarray
            Input array
    :Returns:
        y : ndarray
            copy of x
    :Example:
        >>> import numpy as np
        >>> f = identity
        >>> x = np.array([-100., 50., 10., 40.])
        >>> f(x).tolist()
        [-100.0, 50.0, 10.0, 40.0]

    """
    def __init__(self):
    	self.func_name = "linear"

    def __call__(self, x):
        return 1.0*x

    def inv(self,y):
    	""" Inverse of identity """
    	return 1.0*y

    def deriv(self, x=None, y=None):
        """ Derivative of identity """
        return 1.0
