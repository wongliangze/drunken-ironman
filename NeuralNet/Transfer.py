from __future__ import division
import Error
import numpy as np

# Supported list of transfer functions
transfer_list = ["linear","tanh","logistic","softplus"]

def assign(transfer):
	"""
	    :Parameters:
	    transfer: str
	    	Which transfer function to use
	    	examples: "linear","tanh","logistic","softplus"
	"""	
	if transfer == "linear":
		return identity()
	elif transfer == "tanh":
		return tanh()
	elif transfer == "logistic":
	     return logistic()
	elif transfer == "softplus":
	     return softplus()      
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
    """
    def __init__(self):
    	self.func_name = "linear"

    def __call__(self, x):
        return 1.*x

    def inv(self,y):
    	""" Inverse of identity """
    	return 1.*y

    def deriv(self, x=None, y=None):
        """ Derivative of identity """
        return 1.

class tanh:
    """
    Hyperbolic tangent transfer function

    :Parameters:
        x: ndarray
            Input array
    :Returns:
        y : ndarray
            (1-exp(-2*x))/(1+exp(-2*x))
    """
    def __init__(self):
        self.func_name = "tanh"

    def __call__(self, x):
        return (1.-np.exp(-2.*x))/(1.+np.exp(-2.*x))

    def inv(self,y):
        """ Inverse of tanh """
        assert (1.+y)*(1.-y) > 0, "Saturated tanh.inv computation"
        return 0.5*np.log((1.+y)/(1.-y))

    def deriv(self, x=None, y=None):
        """ Derivative of tanh """
        if y == None:
            if x == None:
                raise TypeError(self.func_name+".deriv requires at least x or y; none given")
            else:
                y = self.__call__(x)    
        return 1. - y**2

class logistic:
    """
    Logistic transfer function

    :Parameters:
        x: ndarray
            Input array
    :Returns:
        y : ndarray
            1/(1+exp(-x))            
    """
    def __init__(self):
        self.func_name = "logistic"

    def __call__(self, x):
        return 1./(1.+np.exp(-x))

    def inv(self,y):
        """ Inverse of logistic """
        assert y*(1.-y) > 0, "Saturated logistic.inv computation"
        return np.log(y/(1.-y))

    def deriv(self, x=None, y=None):
        """ Derivative of logistic """
        if y == None:
            if x == None:
                raise TypeError(self.func_name+".deriv requires at least x or y; none given")
            else:
                y = self.__call__(x)    
        return y*(1. - y)

class softplus:
    """
    Softplus transfer function

    :Parameters:
        x: ndarray
            Input array
    :Returns:
        y : ndarray
            log(1+exp(x))            
    """
    def __init__(self):
        self.func_name = "softplus"

    def __call__(self, x):
        return np.log(1.+np.exp(x))

    def inv(self,y):
        """ Inverse of softplus """
        return np.log(np.exp(y)-1.)

    def deriv(self, x=None, y=None):
        """ Derivative of softplus """
        if x == None:
            if y == None:
                raise TypeError(self.func_name+".deriv requires at least x or y; none given")
            else:
                x = self.inv(y)
        return 1./(1.+np.exp(-x))